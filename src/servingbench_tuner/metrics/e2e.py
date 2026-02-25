from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict
from typing import Any

from servingbench_tuner.client.tracing import RequestTrace
from servingbench_tuner.core.types import E2EMetrics

from .analysis import percentile, pstdev, tail_amplification


def _safe_makespan_s(traces: Sequence[RequestTrace]) -> float:
    if not traces:
        return 0.0
    start = min(t.arrival_s for t in traces)
    end = max(t.end_s for t in traces)
    return max(1e-9, float(end - start))


def aggregate_e2e_metrics(
    traces: list[RequestTrace],
    warmup_requests: int = 0,
) -> tuple[E2EMetrics, dict[str, Any]]:
    """
    Aggregate request traces into production-relevant E2E metrics.

    Definitions (client-side):
    - latency: arrival -> end
    - TTFT: send_start -> first_token
    - TPOT: (end - first_token) / completion_tokens (see tracing.RequestTrace.tpot_s)

    Returns:
      (E2EMetrics, debug_dict)

    Notes:
    - Warmup handling: we drop the first N requests in arrival order.
    - Throughput:
        req/s = ok_requests / makespan
        tok/s = sum(output_tokens over ok) / makespan
      makespan computed on ok traces (post-warmup) if available; else on all post-warmup.
    """
    ordered = sorted(traces, key=lambda t: (t.arrival_s, t.request_id))
    warmup_n = min(max(0, warmup_requests), len(ordered))
    measured = ordered[warmup_n:]

    # Split by status
    ok_traces = [t for t in measured if t.status == "ok"]
    timeout_traces = [t for t in measured if t.status == "timeout"]
    error_traces = [t for t in measured if t.status not in ("ok", "timeout")]

    base_for_span = ok_traces if ok_traces else measured
    makespan = _safe_makespan_s(base_for_span)

    # Latency distributions (ok only, as production often focuses on successful responses)
    lat = [t.latency_s() for t in ok_traces]
    ttft = [t.ttft_s() for t in ok_traces]
    tpot = [t.tpot_s() for t in ok_traces]

    # Token throughput
    ok_reqs = len(ok_traces)
    total_out_tok = sum(int(t.output_tokens) for t in ok_traces)
    rps = float(ok_reqs / makespan) if makespan > 0 else 0.0
    tok_s = float(total_out_tok / makespan) if makespan > 0 else 0.0

    # Rates
    total = max(1, len(measured))
    timeout_rate = float(len(timeout_traces) / total)
    error_rate = float(len(error_traces) / total)
    retry_rate = float(sum(1 for t in measured if (t.retry_count or 0) > 0) / total)

    # Stability
    jitter_std = pstdev(lat)
    tail_amp = tail_amplification(lat, 99.0, 50.0)

    # Tail source breakdown (P95 on ok)
    queue = [float(t.spans.client_queue_s) for t in ok_traces]
    prefill = [float(t.spans.client_prefill_s) for t in ok_traces]
    decode = [float(t.spans.client_decode_s) for t in ok_traces]

    metrics = E2EMetrics(
        ttft_p50=percentile(ttft, 50.0),
        ttft_p95=percentile(ttft, 95.0),
        tpot_p50=percentile(tpot, 50.0),
        tpot_p95=percentile(tpot, 95.0),
        latency_p50=percentile(lat, 50.0),
        latency_p95=percentile(lat, 95.0),
        latency_p99=percentile(lat, 99.0),
        rps=rps,
        tok_s=tok_s,
        timeout_rate=timeout_rate,
        error_rate=error_rate,
        retry_rate=retry_rate,
        jitter_std=jitter_std,
        tail_amp=tail_amp,
        queue_p95=percentile(queue, 95.0),
        prefill_p95=percentile(prefill, 95.0),
        decode_p95=percentile(decode, 95.0),
    )

    debug = {
        "warmup_dropped": warmup_n,
        "measured_total": len(measured),
        "ok": len(ok_traces),
        "timeout": len(timeout_traces),
        "error": len(error_traces),
        "makespan_s": makespan,
        "total_output_tokens_ok": total_out_tok,
        "latency_samples": len(lat),
        "ttft_samples": len(ttft),
        "tpot_samples": len(tpot),
    }
    return metrics, debug


def e2e_metrics_to_dict(metrics: E2EMetrics) -> dict[str, Any]:
    return asdict(metrics)
