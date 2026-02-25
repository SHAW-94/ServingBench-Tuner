from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

# -----------------------------
# Distribution specs
# -----------------------------

DistType = Literal["empirical", "lognormal", "fixed"]


class EmpiricalDistSpec(BaseModel):
    type: Literal["empirical"] = "empirical"
    path: str = Field(..., description="Path to empirical JSON (must include 'values' list)")

    @model_validator(mode="after")
    def _validate_path(self) -> EmpiricalDistSpec:
        if not self.path:
            raise ValueError("empirical dist requires non-empty path")
        return self


class LogNormalDistSpec(BaseModel):
    type: Literal["lognormal"] = "lognormal"
    mu: float = Field(..., description="mu parameter for lognormal")
    sigma: float = Field(..., ge=0.0, description="sigma parameter for lognormal (>=0)")
    min_tokens: int = Field(1, ge=0)
    max_tokens: int = Field(8192, ge=1)

    @model_validator(mode="after")
    def _validate_bounds(self) -> LogNormalDistSpec:
        if self.max_tokens < self.min_tokens:
            raise ValueError("lognormal dist: max_tokens must be >= min_tokens")
        return self


class FixedDistSpec(BaseModel):
    type: Literal["fixed"] = "fixed"
    value: int = Field(..., ge=0)

    @model_validator(mode="after")
    def _validate_value(self) -> FixedDistSpec:
        if self.value < 0:
            raise ValueError("fixed dist: value must be >= 0")
        return self


DistSpec = EmpiricalDistSpec | LogNormalDistSpec | FixedDistSpec


class LengthDistSpec(BaseModel):
    """
    Prompt/output token length distributions.
    """

    prompt: DistSpec
    output: DistSpec


# -----------------------------
# Arrival specs
# -----------------------------

ArrivalMode = Literal["poisson", "burst", "replay"]


class BurstSpec(BaseModel):
    """
    Simple burst model:
      - peak_rps during "on" period
      - base_rps during "off" period
      - cycles repeats (on+off) pattern
    """

    peak_rps: float = Field(..., gt=0.0)
    base_rps: float = Field(0.2, ge=0.0)
    on_s: float = Field(5.0, gt=0.0)
    off_s: float = Field(10.0, ge=0.0)
    cycles: int = Field(10, ge=1)

    @model_validator(mode="after")
    def _validate(self) -> BurstSpec:
        if self.peak_rps < self.base_rps:
            raise ValueError("burst: peak_rps must be >= base_rps")
        return self


class ArrivalSpec(BaseModel):
    mode: ArrivalMode = "poisson"

    # poisson
    rps: float | None = Field(None, gt=0.0, description="Requests per second for Poisson arrivals")

    # burst
    burst: BurstSpec | None = None

    # replay
    trace_path: str | None = Field(None, description="Path to JSONL trace for replay mode")

    @model_validator(mode="after")
    def _validate_mode_fields(self) -> ArrivalSpec:
        if self.mode == "poisson":
            if self.rps is None:
                raise ValueError("arrival.mode=poisson requires arrival.rps")
        elif self.mode == "burst":
            if self.burst is None:
                raise ValueError("arrival.mode=burst requires arrival.burst")
        elif self.mode == "replay" and not self.trace_path:
            raise ValueError("arrival.mode=replay requires arrival.trace_path")
        return self


# -----------------------------
# Session specs
# -----------------------------

ContextGrowth = Literal["append", "window", "agent_like"]


class SessionSpec(BaseModel):
    enabled: bool = False
    turns: int = Field(1, ge=1, le=20)

    # append: context grows by appending history
    # window: keep a moving context window capped by max_context_tokens
    # agent_like: multi-step plan/tool/summarize style; higher variance
    context_growth: ContextGrowth = "append"
    max_context_tokens: int = Field(8192, ge=256)

    # synthetic "think time" between turns (seconds). For replay traces, ignored.
    think_time_s_mu: float = Field(0.2, ge=0.0)
    think_time_s_sigma: float = Field(0.5, ge=0.0)

    @model_validator(mode="after")
    def _validate(self) -> SessionSpec:
        if self.enabled and self.turns < 2:
            # allow turns=1 but it's effectively no-op
            return self
        return self


# -----------------------------
# Workload spec
# -----------------------------


class WorkloadSpec(BaseModel):
    """
    Canonical workload specification (reproducible).
    """

    name: str = Field(..., min_length=1)
    seed: int = Field(42, ge=0)

    arrival: ArrivalSpec
    concurrency_limit: int = Field(64, ge=1, le=100000)
    timeout_s: float = Field(30.0, gt=0.0)
    retries: int = Field(0, ge=0, le=10)

    length_dist: LengthDistSpec
    sessions: SessionSpec = Field(default_factory=SessionSpec)

    warmup_requests: int = Field(50, ge=0)
    duration_s: int = Field(300, ge=1)

    # Optional metadata
    description: str = ""
    tags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate(self) -> WorkloadSpec:
        # Replay mode: duration_s can be ignored; allow
        # Non-replay: duration_s should be long enough for warmup+steady
        if self.arrival.mode != "replay" and self.duration_s < 5:
            raise ValueError("duration_s too small for non-replay workloads (>=5 recommended)")
        return self

    def resolve_paths(self, base_dir: str | Path) -> WorkloadSpec:
        """
        Return a copy with all file paths resolved relative to base_dir.
        """
        base = Path(base_dir).resolve()

        def _resolve(p: str | None) -> str | None:
            if not p:
                return None
            pp = Path(p)
            return str(pp if pp.is_absolute() else (base / pp).resolve())

        obj = self.model_copy(deep=True)
        if obj.arrival.mode == "replay":
            obj.arrival.trace_path = _resolve(obj.arrival.trace_path)
        # length dist paths (empirical)
        if getattr(obj.length_dist.prompt, "type", None) == "empirical":
            obj.length_dist.prompt.path = _resolve(obj.length_dist.prompt.path)  # type: ignore[attr-defined]
        if getattr(obj.length_dist.output, "type", None) == "empirical":
            obj.length_dist.output.path = _resolve(obj.length_dist.output.path)  # type: ignore[attr-defined]
        return obj

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="python")
