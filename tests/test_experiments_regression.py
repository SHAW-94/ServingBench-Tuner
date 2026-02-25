from __future__ import annotations

from tests.helpers import mk_e2e, mk_quality


def test_pct_increase_basic_cases() -> None:
    from servingbench_tuner.experiments.regression import _pct_increase

    assert _pct_increase(110.0, 100.0) == 0.10
    assert _pct_increase(0.0, 0.0) == 0.0
    assert _pct_increase(1.0, 0.0) == 999.0


def test_compare_passes_when_candidate_improves() -> None:
    from servingbench_tuner.experiments.regression import RegressionPolicy, compare

    baseline = mk_e2e(latency_p95=1.0, tok_s=100.0, timeout_rate=0.01, error_rate=0.0)
    candidate = mk_e2e(latency_p95=0.8, tok_s=120.0, timeout_rate=0.01, error_rate=0.0)

    result = compare(
        baseline,
        candidate,
        RegressionPolicy(),
        baseline_quality=mk_quality(0.95),
        candidate_quality=mk_quality(0.96),
    )

    assert result.passed is True
    assert result.reasons == {}
    assert "latency_p95_regress_pct" in result.policy


def test_compare_collects_multiple_fail_reasons() -> None:
    from servingbench_tuner.experiments.regression import RegressionPolicy, compare

    policy = RegressionPolicy(
        latency_p95_regress_pct=0.05,
        tok_s_regress_pct=0.05,
        timeout_rate_increase_abs=0.01,
        error_rate_increase_abs=0.01,
        tail_amp_increase_pct=0.05,
        jitter_std_increase_pct=0.05,
        quality_drop_abs=0.01,
        quality_drop_rel=0.99,
    )
    baseline = mk_e2e(
        latency_p95=1.0,
        tok_s=100.0,
        timeout_rate=0.01,
        error_rate=0.01,
        tail_amp=1.0,
        jitter_std=0.1,
    )
    candidate = mk_e2e(
        latency_p95=1.2,
        tok_s=80.0,
        timeout_rate=0.05,
        error_rate=0.03,
        tail_amp=1.2,
        jitter_std=0.2,
    )

    result = compare(
        baseline,
        candidate,
        policy,
        baseline_quality=mk_quality(0.95),
        candidate_quality=mk_quality(0.85),
    )

    assert result.passed is False
    for key in [
        "latency_p95_regression",
        "tok_s_regression",
        "timeout_rate_increase",
        "error_rate_increase",
        "tail_amp_increase",
        "jitter_std_increase",
        "quality_drop_abs",
        "quality_drop_rel",
    ]:
        assert key in result.reasons
