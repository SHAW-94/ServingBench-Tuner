from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from servingbench_tuner.core.types import GateResult, QualitySummary
from servingbench_tuner.quality.dataset import EvalExample
from servingbench_tuner.quality.evaluators.base import EvalResult, EvaluatorRegistry


def summarize_results(
    results: list[EvalResult],
    details_path: str,
) -> QualitySummary:
    """
    Convert per-example results into QualitySummary.
    overall = average score across all examples (0..1)
    pass_rate = passed fraction
    by_type = average score per example_type
    details_path = where detailed json is stored
    """
    if not results:
        return QualitySummary(overall=0.0, pass_rate=0.0, by_type={}, details_path=details_path)

    overall = sum(r.score for r in results) / max(1, len(results))
    pass_rate = sum(1 for r in results if r.passed) / max(1, len(results))

    by_type: dict[str, list[float]] = {}
    for r in results:
        by_type.setdefault(r.example_type, []).append(float(r.score))
    by_type_avg = {k: (sum(v) / max(1, len(v))) for k, v in by_type.items()}

    return QualitySummary(
        overall=float(overall),
        pass_rate=float(pass_rate),
        by_type={k: float(v) for k, v in by_type_avg.items()},
        details_path=str(details_path),
    )


class QualityGate:
    """
    Production-style quality gate.

    Policy fields (suggested in YAML):
      gate:
        min_overall: 0.70
        min_relative_to_baseline: 0.98     # cand.overall / base.overall
        per_type_min:
          json_schema: 0.98
        hard_fail_types: ["json_schema"]   # if that type avg < 0.999, fail
        fail_fast: true
    """

    def __init__(self, policy: dict[str, Any]) -> None:
        self.policy = policy.get("gate", policy) if isinstance(policy, dict) else {}

        self.min_overall = float(self.policy.get("min_overall", 0.0))
        self.min_rel = float(self.policy.get("min_relative_to_baseline", 0.0))
        self.per_type_min = dict(self.policy.get("per_type_min", {}) or {})
        self.hard_fail_types = list(self.policy.get("hard_fail_types", []) or [])
        self.fail_fast = bool(self.policy.get("fail_fast", False))

    def check(self, cand: QualitySummary, baseline: QualitySummary | None = None) -> GateResult:
        passed = True
        reasons: dict[str, Any] = {}

        if cand.overall < self.min_overall:
            passed = False
            reasons["min_overall"] = {"cand": cand.overall, "min": self.min_overall}

        if baseline is not None and baseline.overall > 1e-12 and self.min_rel > 0:
            rel = cand.overall / baseline.overall
            if rel < self.min_rel:
                passed = False
                reasons["relative_to_baseline"] = {
                    "cand": cand.overall,
                    "base": baseline.overall,
                    "rel": rel,
                    "min_rel": self.min_rel,
                }

        # per-type mins
        for t, v in self.per_type_min.items():
            try:
                thr = float(v)
            except Exception:
                continue
            cand_v = float(cand.by_type.get(t, 0.0))
            if cand_v < thr:
                passed = False
                reasons.setdefault("per_type_min", {})[t] = {"cand": cand_v, "min": thr}

        # hard fail types: typically schema/format tasks must be perfect
        for t in self.hard_fail_types:
            cand_v = float(cand.by_type.get(t, 0.0))
            if cand_v < 0.999:
                passed = False
                reasons.setdefault("hard_fail", {})[t] = {"cand": cand_v, "min": 0.999}

        return GateResult(passed=passed, reasons=reasons)


async def evaluate_pack(
    examples: list[EvalExample],
    outputs_by_id: dict[str, str],
    registry: EvaluatorRegistry,
    details_out_path: str | Path,
    fail_fast: bool = False,
) -> tuple[QualitySummary, list[EvalResult]]:
    """
    Evaluate a whole eval_pack, given model outputs mapping (example_id -> output_text).
    Writes detailed per-example results to details_out_path.

    Returns:
      (QualitySummary, results)
    """
    results: list[EvalResult] = []
    details: list[dict[str, Any]] = []

    for ex in examples:
        if ex.id not in outputs_by_id:
            r = EvalResult(ex.id, ex.type, 0.0, False, {"error": "missing output for example"})
            results.append(r)
            details.append(
                {
                    "id": ex.id,
                    "type": ex.type,
                    "score": r.score,
                    "pass": r.passed,
                    "details": r.details,
                }
            )
            if fail_fast:
                break
            continue

        ev = registry.route(ex)
        out = outputs_by_id.get(ex.id, "")
        r = await ev.evaluate(ex, out)

        # normalize score into [0,1]
        r.score = float(max(0.0, min(1.0, r.score)))
        results.append(r)
        details.append(
            {"id": ex.id, "type": ex.type, "score": r.score, "pass": r.passed, "details": r.details}
        )

        if fail_fast and (not r.passed):
            break

    # write details
    p = Path(details_out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"results": details}, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = summarize_results(results, str(p))
    return summary, results


def write_quality_summary(
    summary: QualitySummary,
    gate: GateResult | None,
    out_path: str | Path,
) -> None:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    obj: dict[str, Any] = {
        "overall": summary.overall,
        "pass_rate": summary.pass_rate,
        "by_type": summary.by_type,
        "details_path": summary.details_path,
    }
    if gate is not None:
        obj["gate"] = asdict(gate)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
