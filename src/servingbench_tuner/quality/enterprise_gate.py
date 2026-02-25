from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class EnterpriseQualityGateConfig:
    min_overall_relative: float = 0.98
    min_qa_relative: float = 0.99
    min_structured_relative: float = 0.99
    min_code_relative: float = 0.97
    min_summary_relative: float = 0.98
    min_structured_absolute: float = 0.98


@dataclass
class EnterpriseQualityGateResult:
    passed: bool
    reasons: dict[str, str]
    ratios: dict[str, float]


def _get_breakdown(row: dict[str, Any]) -> dict[str, float]:
    q = row.get("quality_breakdown") or {}
    return {
        "qa": float(q.get("qa", 0.0)),
        "structured": float(q.get("structured", 0.0)),
        "code": float(q.get("code", 0.0)),
        "summary": float(q.get("summary", 0.0)),
    }


def evaluate_enterprise_quality_gate(
    target_row: dict[str, Any],
    baseline_row: dict[str, Any],
    cfg: EnterpriseQualityGateConfig,
) -> EnterpriseQualityGateResult:
    reasons: dict[str, str] = {}
    ratios: dict[str, float] = {}

    t_overall = float(target_row.get("quality", 0.0))
    b_overall = max(1e-9, float(baseline_row.get("quality", 0.0)))
    ratios["overall"] = t_overall / b_overall
    if ratios["overall"] < cfg.min_overall_relative:
        reasons["overall"] = f"{ratios['overall']:.4f} < {cfg.min_overall_relative:.4f}"

    tb = _get_breakdown(target_row)
    bb = _get_breakdown(baseline_row)

    for key, rel_min in [
        ("qa", cfg.min_qa_relative),
        ("structured", cfg.min_structured_relative),
        ("code", cfg.min_code_relative),
        ("summary", cfg.min_summary_relative),
    ]:
        base = max(1e-9, bb.get(key, 0.0))
        ratio = tb.get(key, 0.0) / base
        ratios[key] = ratio
        if ratio < rel_min:
            reasons[key] = f"{ratio:.4f} < {rel_min:.4f}"

    if tb.get("structured", 0.0) < cfg.min_structured_absolute:
        reasons["structured_abs"] = (
            f"{tb.get('structured', 0.0):.4f} < {cfg.min_structured_absolute:.4f}"
        )

    return EnterpriseQualityGateResult(passed=not reasons, reasons=reasons, ratios=ratios)
