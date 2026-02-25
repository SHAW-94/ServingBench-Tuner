from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from servingbench_tuner.core.types import QualitySummary
from servingbench_tuner.quality.dataset import EvalExample
from servingbench_tuner.quality.evaluators.base import EvaluatorRegistry
from servingbench_tuner.quality.gate import evaluate_pack


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_baseline(path: str | Path) -> QualitySummary:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"baseline file not found: {p}")
    obj = json.loads(p.read_text(encoding="utf-8"))
    return QualitySummary(
        overall=float(obj.get("overall", 0.0)),
        pass_rate=float(obj.get("pass_rate", 0.0)),
        by_type={str(k): float(v) for k, v in (obj.get("by_type", {}) or {}).items()},
        details_path=str(obj.get("details_path", "")),
    )


def save_baseline(
    path: str | Path, summary: QualitySummary, meta: dict[str, Any] | None = None
) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    obj: dict[str, Any] = {
        "created_at": _utc_now(),
        "overall": summary.overall,
        "pass_rate": summary.pass_rate,
        "by_type": summary.by_type,
        "details_path": summary.details_path,
    }
    if meta:
        obj["meta"] = meta
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


async def generate_baseline_from_outputs(
    examples: list[EvalExample],
    outputs_by_id: dict[str, str],
    registry: EvaluatorRegistry,
    details_out_path: str | Path,
) -> QualitySummary:
    """
    Build baseline summary from existing model outputs.
    """
    summary, _ = await evaluate_pack(
        examples=examples,
        outputs_by_id=outputs_by_id,
        registry=registry,
        details_out_path=details_out_path,
        fail_fast=False,
    )
    return summary


async def generate_baseline_via_infer_fn(
    examples: list[EvalExample],
    infer_fn: Callable[[EvalExample], str],
    registry: EvaluatorRegistry,
    details_out_path: str | Path,
) -> tuple[QualitySummary, dict[str, str]]:
    """
    Generate baseline by calling a provided infer_fn(example)->output_text (sync callable).
    Useful when you want to compute baseline once and store it.

    Returns:
      (summary, outputs_by_id)
    """
    outputs: dict[str, str] = {}
    for ex in examples:
        outputs[ex.id] = infer_fn(ex)

    summary = await generate_baseline_from_outputs(
        examples=examples,
        outputs_by_id=outputs,
        registry=registry,
        details_out_path=details_out_path,
    )
    return summary, outputs


def ensure_baseline(
    baseline_path: str | Path,
    build_fn: Callable[[], QualitySummary],
) -> QualitySummary:
    """
    Load baseline if exists; otherwise build it (build_fn) and save.

    build_fn is sync, so you can wrap an async baseline generation with anyio.run externally.
    """
    p = Path(baseline_path)
    if p.exists():
        return load_baseline(p)
    summary = build_fn()
    save_baseline(p, summary, meta={"note": "auto-generated baseline"})
    return summary
