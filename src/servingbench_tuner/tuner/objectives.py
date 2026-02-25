from __future__ import annotations

from dataclasses import dataclass

from servingbench_tuner.core.types import E2EMetrics, QualitySummary, ServerMetrics


@dataclass
class ObjectiveConfig:
    """
    Multi-objective configuration.

    We optimize under constraints; objectives define Pareto trade-offs.

    Default objectives:
      - minimize latency_p95
      - minimize cost_proxy
      - maximize tok_s (implemented as minimize -tok_s)
    """

    minimize: list[str] = None  # names of objectives to minimize
    maximize: list[str] = None  # names to maximize (converted to negative)

    # cost proxy weights
    cost_vram_weight: float = 1.0
    cost_util_weight: float = 0.0  # optional; use if you trust util

    def __post_init__(self) -> None:
        if self.minimize is None:
            self.minimize = ["latency_p95_s", "cost_proxy"]
        if self.maximize is None:
            self.maximize = ["tok_s"]


def compute_cost_proxy(
    e2e: E2EMetrics,
    server: ServerMetrics | None,
    cfg: ObjectiveConfig,
) -> float:
    """
    A pragmatic cost proxy:
      cost_proxy = (vram_peak_mb * w1) / max(tok_s, eps)

    Intuition:
      - Higher VRAM often correlates with larger/expensive GPU footprint
      - tokens/s correlates with how many GPUs you need for a target load

    If server metrics are absent (CPU/CI), fall back to:
      cost_proxy = 1 / max(tok_s, eps) scaled by latency
    """
    eps = 1e-9
    tok_s = max(eps, float(e2e.tok_s))
    if server is None or float(server.vram_peak_mb) <= 0:
        # CPU/CI fallback
        return float((1.0 / tok_s) * (1.0 + float(e2e.latency_p95) * 0.1))

    vram = float(server.vram_peak_mb)
    # optional: if gpu_util_avg exists, higher util might mean better cost efficiency,
    # but it's noisy; default weight 0.
    util_term = 0.0
    if server.gpu_util_avg is not None and cfg.cost_util_weight > 0:
        util = max(1e-6, float(server.gpu_util_avg) / 100.0)
        util_term = cfg.cost_util_weight * (1.0 / util)

    return float((cfg.cost_vram_weight * vram + util_term) / tok_s)


def extract_objective_values(
    e2e: E2EMetrics,
    server: ServerMetrics | None,
    quality: QualitySummary | None,
    cfg: ObjectiveConfig,
) -> dict[str, float]:
    """
    Produce a flat dict of objective values.
    """
    d: dict[str, float] = {
        "latency_p95_s": float(e2e.latency_p95),
        "latency_p99_s": float(e2e.latency_p99),
        "ttft_p95_s": float(e2e.ttft_p95),
        "tok_s": float(e2e.tok_s),
        "rps": float(e2e.rps),
        "timeout_rate": float(e2e.timeout_rate),
        "error_rate": float(e2e.error_rate),
        "jitter_std": float(e2e.jitter_std),
        "tail_amp": float(e2e.tail_amp),
    }

    d["cost_proxy"] = compute_cost_proxy(e2e, server, cfg)

    if server is not None:
        d["vram_peak_mb"] = float(server.vram_peak_mb)
        d["vram_avg_mb"] = float(server.vram_avg_mb)
        d["vram_fragmentation_ratio"] = float(server.vram_fragmentation_ratio)
        if server.gpu_util_avg is not None:
            d["gpu_util_avg"] = float(server.gpu_util_avg)

    if quality is not None:
        d["quality_overall"] = float(quality.overall)
        d["quality_pass_rate"] = float(quality.pass_rate)

    return d


def objective_vector(
    values: dict[str, float],
    cfg: ObjectiveConfig,
) -> tuple[list[float], list[str]]:
    """
    Convert objective dict into numeric vector for multi-objective optimization.

    - minimize objectives: keep as-is
    - maximize objectives: negate so that we still minimize
    Returns (vec, names)
    """
    vec: list[float] = []
    names: list[str] = []

    for k in cfg.minimize or []:
        vec.append(float(values.get(k, 0.0)))
        names.append(k)

    for k in cfg.maximize or []:
        vec.append(float(-values.get(k, 0.0)))
        names.append(f"maximize:{k}")

    return vec, names
