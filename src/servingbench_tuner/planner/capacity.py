from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class CapacityInput:
    """
    Capacity planning input.

    Required:
      - peak_rps: peak requests per second
      - p95_sla_s: SLA threshold for P95 latency (seconds)
      - avg_output_tokens: expected average output tokens per request

    Recommended:
      - avg_input_tokens: affects prefill cost and memory; used for notes only here
      - tok_s_per_gpu: measured tokens/s per GPU from benchmark at similar workload
      - target_util: utilization target (e.g., 0.65 ~ 0.85)
      - gpu_hourly_cost: for cost estimate (optional)
    """

    peak_rps: float
    p95_sla_s: float
    avg_output_tokens: float

    avg_input_tokens: float = 0.0

    tok_s_per_gpu: float = 0.0
    target_util: float = 0.75

    gpu_hourly_cost: float = 0.0


@dataclass
class CapacityPlan:
    """
    Capacity planning output.
    """

    required_gpus: int
    recommended_concurrency_limit: int
    peak_tok_s: float
    effective_tok_s_per_gpu: float
    cost_per_hour: float | None
    notes: dict[str, Any]


def plan_capacity(inp: CapacityInput) -> CapacityPlan:
    """
    Production-ish capacity planner (simple, explainable):

    1) concurrency (Little's law proxy):
       concurrency_needed ≈ peak_rps * p95_sla_s
       -> set concurrency_limit slightly above

    2) token throughput demand:
       peak_tok_s ≈ peak_rps * avg_output_tokens

    3) required GPUs:
       gpus = ceil(peak_tok_s / (tok_s_per_gpu * target_util))

    If tok_s_per_gpu is unknown (0), we still output concurrency recommendation,
    and set required_gpus to 0 with note to measure tok_s_per_gpu via benchmark.
    """
    peak_rps = max(0.0, float(inp.peak_rps))
    sla = max(1e-6, float(inp.p95_sla_s))
    out_tok = max(0.0, float(inp.avg_output_tokens))

    concurrency_needed = peak_rps * sla
    # add headroom (tail + burst)
    recommended_conc = int(math.ceil(concurrency_needed * 1.3))
    recommended_conc = max(1, recommended_conc)

    peak_tok_s = peak_rps * out_tok

    tok_s_gpu = max(0.0, float(inp.tok_s_per_gpu))
    util = min(0.99, max(0.10, float(inp.target_util)))
    effective = tok_s_gpu * util

    if effective <= 1e-9:
        notes = {
            "needs_measurement": True,
            "hint": "tok_s_per_gpu 未提供或为 0：请先在目标 workload 上跑 benchmark 得到 tok/s，然后再估算 GPU 数。",
            "concurrency_formula": "concurrency ≈ peak_rps * p95_sla_s * 1.3",
            "token_demand_formula": "peak_tok_s ≈ peak_rps * avg_output_tokens",
        }
        return CapacityPlan(
            required_gpus=0,
            recommended_concurrency_limit=recommended_conc,
            peak_tok_s=float(peak_tok_s),
            effective_tok_s_per_gpu=float(effective),
            cost_per_hour=None,
            notes=notes,
        )

    required = int(math.ceil(peak_tok_s / effective))
    required = max(1, required)

    cost = None
    if inp.gpu_hourly_cost and inp.gpu_hourly_cost > 0:
        cost = float(required * float(inp.gpu_hourly_cost))

    notes = {
        "needs_measurement": False,
        "concurrency_needed_est": float(concurrency_needed),
        "recommended_headroom": "30%",
        "util_target": util,
        "avg_input_tokens_note": float(inp.avg_input_tokens),
        "assumptions": [
            "tok_s_per_gpu 来自与目标负载相似的 benchmark；差异会直接影响估算",
            "SLA 使用 p95_sla_s 做近似，实际还需考虑队列拥塞与超时/重试策略",
        ],
    }

    return CapacityPlan(
        required_gpus=required,
        recommended_concurrency_limit=recommended_conc,
        peak_tok_s=float(peak_tok_s),
        effective_tok_s_per_gpu=float(effective),
        cost_per_hour=cost,
        notes=notes,
    )


def plan_to_dict(plan: CapacityPlan) -> dict[str, Any]:
    return asdict(plan)
