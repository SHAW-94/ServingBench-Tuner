from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from servingbench_tuner.core.types import E2EMetrics, QualitySummary


@dataclass
class RegressionPolicy:
    """
    Regression thresholds for CI/production-ish gating.

    - latency_p95_regress_pct: if candidate P95 is worse than baseline by > X%, fail
    - tok_s_regress_pct: if tok/s drops by > X%, fail
    - timeout_rate_increase_abs: if timeout_rate increases by > abs threshold, fail
    - quality_drop_abs: if quality.overall drops by > abs threshold, fail
    - quality_drop_rel: if candidate/baseline drops below this ratio, fail
    """

    latency_p95_regress_pct: float = 0.10
    tok_s_regress_pct: float = 0.10
    timeout_rate_increase_abs: float = 0.02
    error_rate_increase_abs: float = 0.01

    quality_drop_abs: float = 0.02
    quality_drop_rel: float = 0.98

    # Optional stability regression
    tail_amp_increase_pct: float = 0.20
    jitter_std_increase_pct: float = 0.20


@dataclass
class RegressionResult:
    passed: bool
    reasons: dict[str, Any]
    policy: dict[str, Any]


def _pct_increase(new: float, old: float) -> float:
    if old <= 1e-12:
        return 0.0 if new <= 1e-12 else 999.0
    return (new - old) / old


def compare(
    baseline_e2e: E2EMetrics,
    candidate_e2e: E2EMetrics,
    policy: RegressionPolicy,
    baseline_quality: QualitySummary | None = None,
    candidate_quality: QualitySummary | None = None,
) -> RegressionResult:
    """
    Compare candidate against baseline and decide pass/fail.
    """
    reasons: dict[str, Any] = {}
    passed = True

    # Latency regression (P95)
    p95_inc = _pct_increase(float(candidate_e2e.latency_p95), float(baseline_e2e.latency_p95))
    if p95_inc > policy.latency_p95_regress_pct:
        passed = False
        reasons["latency_p95_regression"] = {
            "baseline": float(baseline_e2e.latency_p95),
            "candidate": float(candidate_e2e.latency_p95),
            "increase_pct": p95_inc,
            "threshold_pct": policy.latency_p95_regress_pct,
        }

    # Throughput regression (tok/s)
    # tok_drop = _pct_increase(float(baseline_e2e.tok_s), float(candidate_e2e.tok_s))
    # tok_drop is baseline->candidate "increase", so if negative means candidate higher.
    # We want to fail if candidate tok_s is lower by > X%:
    if float(candidate_e2e.tok_s) > 1e-12:
        tok_s_inc = _pct_increase(float(candidate_e2e.tok_s), float(baseline_e2e.tok_s))
    else:
        tok_s_inc = -1.0
    if tok_s_inc < -policy.tok_s_regress_pct:
        passed = False
        reasons["tok_s_regression"] = {
            "baseline": float(baseline_e2e.tok_s),
            "candidate": float(candidate_e2e.tok_s),
            "decrease_pct": -tok_s_inc,
            "threshold_pct": policy.tok_s_regress_pct,
        }

    # Timeout/error regression (absolute increase)
    timeout_inc = float(candidate_e2e.timeout_rate) - float(baseline_e2e.timeout_rate)
    if timeout_inc > policy.timeout_rate_increase_abs:
        passed = False
        reasons["timeout_rate_increase"] = {
            "baseline": float(baseline_e2e.timeout_rate),
            "candidate": float(candidate_e2e.timeout_rate),
            "increase_abs": timeout_inc,
            "threshold_abs": policy.timeout_rate_increase_abs,
        }

    err_inc = float(candidate_e2e.error_rate) - float(baseline_e2e.error_rate)
    if err_inc > policy.error_rate_increase_abs:
        passed = False
        reasons["error_rate_increase"] = {
            "baseline": float(baseline_e2e.error_rate),
            "candidate": float(candidate_e2e.error_rate),
            "increase_abs": err_inc,
            "threshold_abs": policy.error_rate_increase_abs,
        }

    # Stability regression (tail amp/jitter)
    tail_inc = _pct_increase(float(candidate_e2e.tail_amp), float(baseline_e2e.tail_amp))
    if tail_inc > policy.tail_amp_increase_pct:
        passed = False
        reasons["tail_amp_increase"] = {
            "baseline": float(baseline_e2e.tail_amp),
            "candidate": float(candidate_e2e.tail_amp),
            "increase_pct": tail_inc,
            "threshold_pct": policy.tail_amp_increase_pct,
        }

    jitter_inc = _pct_increase(float(candidate_e2e.jitter_std), float(baseline_e2e.jitter_std))
    if jitter_inc > policy.jitter_std_increase_pct:
        passed = False
        reasons["jitter_std_increase"] = {
            "baseline": float(baseline_e2e.jitter_std),
            "candidate": float(candidate_e2e.jitter_std),
            "increase_pct": jitter_inc,
            "threshold_pct": policy.jitter_std_increase_pct,
        }

    # Quality regression
    if baseline_quality is not None and candidate_quality is not None:
        q_drop_abs = float(baseline_quality.overall) - float(candidate_quality.overall)
        if q_drop_abs > policy.quality_drop_abs:
            passed = False
            reasons["quality_drop_abs"] = {
                "baseline": float(baseline_quality.overall),
                "candidate": float(candidate_quality.overall),
                "drop_abs": q_drop_abs,
                "threshold_abs": policy.quality_drop_abs,
            }

        if float(baseline_quality.overall) > 1e-12:
            rel = float(candidate_quality.overall) / float(baseline_quality.overall)
            if rel < policy.quality_drop_rel:
                passed = False
                reasons["quality_drop_rel"] = {
                    "baseline": float(baseline_quality.overall),
                    "candidate": float(candidate_quality.overall),
                    "ratio": rel,
                    "min_ratio": policy.quality_drop_rel,
                }

    return RegressionResult(passed=passed, reasons=reasons, policy=asdict(policy))
