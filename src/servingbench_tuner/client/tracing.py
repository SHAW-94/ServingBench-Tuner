from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Status = Literal["ok", "timeout", "error"]


@dataclass
class SpanBreakdown:
    """
    Span breakdown for diagnosing tail latency.
    Two layers:
    - client_* : measured at client side (includes client-side concurrency gating)
    - server_* : if backend provides (mock can, real server may not)
    """

    client_queue_s: float = 0.0  # waiting for client concurrency slot
    client_prefill_s: float = 0.0  # approx: (first_token - send_start)
    client_decode_s: float = 0.0  # approx: (end - first_token)

    server_queue_s: float | None = None
    server_prefill_s: float | None = None
    server_decode_s: float | None = None


@dataclass
class TokenTiming:
    """
    Timestamp of each streamed token/chunk at client side.
    `t_s` uses time.time() wall-clock for simplicity.
    """

    index: int
    t_s: float


@dataclass
class RequestTrace:
    """
    Canonical request trace record.

    Time origin:
      - arrival_s is relative to run_start_t (seconds since run start)
      - other *_s are also relative to run_start_t
    """

    request_id: str
    arrival_s: float
    send_start_s: float
    first_token_s: float
    end_s: float

    input_tokens: int
    output_tokens: int

    status: Status = "ok"
    error: str = ""
    retry_count: int = 0

    session_id: str | None = None
    turn_id: int | None = None

    spans: SpanBreakdown = field(default_factory=SpanBreakdown)

    # useful for debugging/metrics
    token_timings: list[TokenTiming] = field(default_factory=list)

    # optional metadata
    meta: dict[str, Any] = field(default_factory=dict)

    # -----------------------
    # Metric definitions
    # -----------------------
    def latency_s(self) -> float:
        """Request-level latency: from arrival to end."""
        return max(0.0, self.end_s - self.arrival_s)

    def ttft_s(self) -> float:
        """
        TTFT definition (client-side):
        from *send_start* to *first token received*.
        If first_token_s <= send_start_s, returns 0.
        """
        return max(0.0, self.first_token_s - self.send_start_s)

    def tpot_s(self) -> float:
        """
        TPOT definition (client-side):
        average time per output token during decode phase.

        Preferred:
          (end - first_token) / completion_tokens
        If completion_tokens is unknown, fallback to:
          (end - first_token) / max(1, (num_token_events))
        """
        decode = max(0.0, self.end_s - self.first_token_s)
        denom = max(1, int(self.output_tokens))
        return decode / denom

    def jitter_anchor_latency_s(self) -> float:
        """Useful for stability metric (std)."""
        return self.latency_s()

    def tail_components(self) -> dict[str, float]:
        """
        For debugging: where did P95/P99 come from?
        """
        return {
            "client_queue_s": self.spans.client_queue_s,
            "client_prefill_s": self.spans.client_prefill_s,
            "client_decode_s": self.spans.client_decode_s,
        }

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # dataclasses nested already converted
        return d


def now_s() -> float:
    return time.time()
