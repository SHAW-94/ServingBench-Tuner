from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

# ---- Workload / Serving Specs ----


@dataclass
class WorkloadSpec:
    name: str
    seed: int = 42
    arrival_mode: Literal["poisson", "burst", "replay"] = "replay"
    trace_path: str | None = None

    concurrency_limit: int = 64
    timeout_s: float = 30.0
    retries: int = 0

    # empirical length dist json paths (optional)
    prompt_len_path: str | None = None
    output_len_path: str | None = None

    sessions_enabled: bool = False
    turns: int = 1
    context_growth: Literal["append", "window", "agent_like"] = "append"
    max_context_tokens: int = 8192

    warmup_requests: int = 50
    duration_s: int = 300


@dataclass
class ServingConfig:
    # For the minimal mock backend, only a few knobs matter.
    # vLLM config can be stored and passed through as well.
    params: dict[str, Any] = field(default_factory=dict)


# ---- Request trace ----


@dataclass
class RequestTrace:
    request_id: str
    arrival_s: float
    start_s: float
    first_token_s: float
    end_s: float

    input_tokens: int
    output_tokens: int

    status: Literal["ok", "timeout", "error"] = "ok"
    retry_count: int = 0
    session_id: str | None = None
    turn_id: int | None = None

    queue_time_s: float = 0.0
    prefill_time_s: float = 0.0
    decode_time_s: float = 0.0

    @property
    def ttft_s(self) -> float:
        return max(0.0, self.first_token_s - self.arrival_s)

    @property
    def latency_s(self) -> float:
        return max(0.0, self.end_s - self.arrival_s)

    @property
    def tpot_s(self) -> float:
        # time-per-output-token (approx). Avoid div-by-zero.
        if self.output_tokens <= 0:
            return 0.0
        return max(0.0, self.decode_time_s / max(1, self.output_tokens))


# ---- Metrics ----


@dataclass
class E2EMetrics:
    ttft_p50: float
    ttft_p95: float
    tpot_p50: float
    tpot_p95: float
    latency_p50: float
    latency_p95: float
    latency_p99: float
    rps: float
    tok_s: float
    timeout_rate: float
    error_rate: float
    retry_rate: float
    jitter_std: float
    tail_amp: float  # e.g. p99/p50
    queue_p95: float
    prefill_p95: float
    decode_p95: float


@dataclass
class ServerMetrics:
    vram_peak_mb: float
    vram_avg_mb: float
    vram_fragmentation_ratio: float
    gpu_util_avg: float | None = None


@dataclass
class QualitySummary:
    overall: float
    pass_rate: float
    by_type: dict[str, float]
    details_path: str


@dataclass
class GateResult:
    passed: bool
    reasons: dict[str, Any] = field(default_factory=dict)


# ---- Storage record ----


@dataclass
class RunRecord:
    run_id: str
    created_at: str
    kind: Literal["benchmark", "tune"]
    backend: str
    workload_path: str
    serving_path: str
    quality_path: str
    tuner_path: str
    env_json: dict[str, Any]
    metrics_json: dict[str, Any]
    server_json: dict[str, Any]
    quality_json: dict[str, Any]
    artifacts: dict[str, str] = field(default_factory=dict)
