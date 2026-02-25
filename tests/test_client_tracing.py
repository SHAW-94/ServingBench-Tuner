from __future__ import annotations

import pytest


def test_request_trace_metrics_and_tail_components() -> None:
    from servingbench_tuner.client.tracing import RequestTrace, SpanBreakdown, TokenTiming

    trace = RequestTrace(
        request_id="r1",
        arrival_s=1.0,
        send_start_s=1.2,
        first_token_s=1.5,
        end_s=2.1,
        input_tokens=10,
        output_tokens=3,
        spans=SpanBreakdown(client_queue_s=0.2, client_prefill_s=0.3, client_decode_s=0.6),
        token_timings=[TokenTiming(index=0, t_s=1.5)],
        meta={"k": "v"},
    )

    assert trace.latency_s() == 1.1
    assert trace.ttft_s() == pytest.approx(0.3)
    assert trace.tpot_s() == pytest.approx(0.2)
    assert trace.jitter_anchor_latency_s() == trace.latency_s()
    assert trace.tail_components() == {
        "client_queue_s": 0.2,
        "client_prefill_s": 0.3,
        "client_decode_s": 0.6,
    }

    payload = trace.to_dict()
    assert payload["request_id"] == "r1"
    assert payload["spans"]["client_queue_s"] == 0.2
    assert payload["meta"]["k"] == "v"


def test_request_trace_metrics_are_clamped_non_negative() -> None:
    from servingbench_tuner.client.tracing import RequestTrace

    trace = RequestTrace(
        request_id="r2",
        arrival_s=5.0,
        send_start_s=5.0,
        first_token_s=4.0,
        end_s=3.5,
        input_tokens=1,
        output_tokens=0,
    )

    assert trace.latency_s() == 0.0
    assert trace.ttft_s() == 0.0
    assert trace.tpot_s() == 0.0
