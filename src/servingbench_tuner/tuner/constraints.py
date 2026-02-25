from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from servingbench_tuner.core.types import E2EMetrics, QualitySummary, ServerMetrics


@dataclass
class ConstraintConfig:
    """Hard constraints for feasibility checks during tuning."""

    min_quality_overall: float | None = None
    min_quality_relative: float | None = None  # candidate >= baseline * this ratio

    vram_limit_mb: float | None = None
    vram_safety_margin_ratio: float = 0.0  # e.g. 0.05 => enforce peak <= 95% of vram_limit_mb

    max_timeout_rate: float | None = None
    max_error_rate: float | None = None
    max_tail_amp: float | None = None
    max_ttft_p95_s: float | None = None


def _f(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def check_constraints(
    e2e: E2EMetrics,
    server: ServerMetrics | None,
    quality: QualitySummary | None,
    cfg: ConstraintConfig,
    baseline_quality: QualitySummary | None = None,
) -> tuple[bool, dict[str, Any]]:
    """
    Returns (feasible, violations).

    violations uses stable keys so reports can surface what failed.
    """
    violations: dict[str, Any] = {}

    # --- Quality ---
    if cfg.min_quality_overall is not None:
        q = _f(getattr(quality, "overall", None), -1.0)
        if q < float(cfg.min_quality_overall):
            violations["min_quality_overall"] = {
                "candidate": q,
                "min": float(cfg.min_quality_overall),
            }

    if cfg.min_quality_relative is not None and baseline_quality is not None:
        cand_q = _f(getattr(quality, "overall", None), -1.0)
        base_q = _f(getattr(baseline_quality, "overall", None), 0.0)
        min_ratio = float(cfg.min_quality_relative)
        required = base_q * min_ratio
        if cand_q < required:
            violations["min_quality_relative"] = {
                "candidate": cand_q,
                "baseline": base_q,
                "min_ratio": min_ratio,
                "required": required,
            }

    # --- VRAM ---
    if cfg.vram_limit_mb is not None:
        peak = _f(getattr(server, "vram_peak_mb", None), 0.0)
        raw_limit = float(cfg.vram_limit_mb)
        margin_ratio = max(0.0, min(0.5, float(cfg.vram_safety_margin_ratio or 0.0)))
        effective_limit = raw_limit * (1.0 - margin_ratio)
        if peak > effective_limit:
            violations["vram_limit_mb"] = {
                "candidate": peak,
                "limit": raw_limit,
                "effective_limit": effective_limit,
                "safety_margin_ratio": margin_ratio,
            }

    # --- Reliability / latency guardrails ---
    if cfg.max_timeout_rate is not None:
        timeout_rate = _f(getattr(e2e, "timeout_rate", None), 0.0)
        if timeout_rate > float(cfg.max_timeout_rate):
            violations["max_timeout_rate"] = {
                "candidate": timeout_rate,
                "max": float(cfg.max_timeout_rate),
            }

    if cfg.max_error_rate is not None:
        error_rate = _f(getattr(e2e, "error_rate", None), 0.0)
        if error_rate > float(cfg.max_error_rate):
            violations["max_error_rate"] = {
                "candidate": error_rate,
                "max": float(cfg.max_error_rate),
            }

    if cfg.max_tail_amp is not None:
        tail_amp = _f(getattr(e2e, "tail_amp", None), 0.0)
        if tail_amp > float(cfg.max_tail_amp):
            violations["max_tail_amp"] = {
                "candidate": tail_amp,
                "max": float(cfg.max_tail_amp),
            }

    if cfg.max_ttft_p95_s is not None:
        ttft_p95 = _f(getattr(e2e, "ttft_p95", None), 0.0)
        if ttft_p95 > float(cfg.max_ttft_p95_s):
            violations["max_ttft_p95_s"] = {
                "candidate": ttft_p95,
                "max": float(cfg.max_ttft_p95_s),
            }

    return (len(violations) == 0), violations
