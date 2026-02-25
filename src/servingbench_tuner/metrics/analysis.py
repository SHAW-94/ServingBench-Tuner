from __future__ import annotations

import math
import statistics
from collections.abc import Iterable, Sequence


def percentile(values: Sequence[float], p: float) -> float:
    """
    Continuous percentile (linear interpolation) similar to numpy.quantile(method='linear').
    p in [0, 100].
    """
    if not values:
        return 0.0
    if p <= 0:
        return float(min(values))
    if p >= 100:
        return float(max(values))
    v = sorted(float(x) for x in values)
    k = (len(v) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(v[int(k)])
    d0 = v[f] * (c - k)
    d1 = v[c] * (k - f)
    return float(d0 + d1)


def mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def pstdev(values: Sequence[float]) -> float:
    """
    Population standard deviation (stable for small samples).
    """
    if len(values) <= 1:
        return 0.0
    return float(statistics.pstdev(values))


def quantile_gap(values: Sequence[float], low_p: float, high_p: float) -> float:
    """
    High-low quantile difference (e.g., p95 - p50) as a "jitter magnitude".
    """
    return max(0.0, percentile(values, high_p) - percentile(values, low_p))


def tail_amplification(
    values: Sequence[float], tail_p: float = 99.0, anchor_p: float = 50.0
) -> float:
    """
    Tail amplification ratio, e.g. p99/p50. Returns 0 if anchor is 0.
    """
    anchor = percentile(values, anchor_p)
    tail = percentile(values, tail_p)
    if anchor <= 1e-12:
        return 0.0
    return float(tail / anchor)


def stability_summary(values: Sequence[float]) -> dict[str, float]:
    """
    Basic stability metrics used in production-ish discussions:
    - jitter_std: std of request latency
    - p95_minus_p50: quantile gap
    - tail_amp_p99_p50: p99/p50
    """
    return {
        "jitter_std": pstdev(values),
        "p95_minus_p50": quantile_gap(values, 50.0, 95.0),
        "tail_amp_p99_p50": tail_amplification(values, 99.0, 50.0),
    }


def split_ok_timeout_error(statuses: Iterable[str]) -> tuple[int, int, int]:
    ok = timeout = error = 0
    for s in statuses:
        if s == "ok":
            ok += 1
        elif s == "timeout":
            timeout += 1
        else:
            error += 1
    return ok, timeout, error
