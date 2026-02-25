from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

ParamType = Literal["int", "float", "categorical", "bool"]


@dataclass(frozen=True)
class ParamSpec:
    name: str
    ptype: ParamType
    default: Any
    description: str = ""

    # bounds
    low: float | None = None
    high: float | None = None

    # categories
    choices: list[Any] | None = None

    # step for ints
    step: int | None = None

    def validate(self, v: Any) -> tuple[bool, str]:
        if self.ptype == "int":
            if not isinstance(v, int):
                return False, f"{self.name} must be int"
            if self.low is not None and v < int(self.low):
                return False, f"{self.name} < low"
            if self.high is not None and v > int(self.high):
                return False, f"{self.name} > high"
            if self.step and self.step > 1:
                base = int(self.low) if self.low is not None else 0
                if ((v - base) % self.step) != 0:
                    return False, f"{self.name} must align to step={self.step}"
            return True, ""
        if self.ptype == "float":
            if not isinstance(v, int | float):
                return False, f"{self.name} must be float"
            vv = float(v)
            if self.low is not None and vv < float(self.low):
                return False, f"{self.name} < low"
            if self.high is not None and vv > float(self.high):
                return False, f"{self.name} > high"
            return True, ""
        if self.ptype == "categorical":
            if self.choices is None or not self.choices:
                return False, f"{self.name} missing choices"
            if v not in self.choices:
                return False, f"{self.name} not in choices"
            return True, ""
        if self.ptype == "bool":
            if not isinstance(v, bool):
                return False, f"{self.name} must be bool"
            return True, ""
        return False, f"unknown ptype {self.ptype}"


@dataclass
class Candidate:
    """
    A candidate configuration in the search space.
    """

    params: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return dict(self.params)

    def get(self, k: str, default: Any = None) -> Any:
        return self.params.get(k, default)


@dataclass
class SearchSpace:
    """
    Define 8~12 parameters for serving / client behavior.

    Notes:
    - Some are "server-start-time" parameters (e.g., kv_cache_dtype, quantization).
      In a real deployment, you'd output a recommended config card and restart server.
    - Some are "client-side" parameters (e.g., concurrency_limit, request_timeout_s).
      Those can be applied immediately in your runner.

    The orchestrator can decide what to hot-apply vs. what to just report.
    """

    specs: list[ParamSpec] = field(default_factory=list)

    def spec_map(self) -> dict[str, ParamSpec]:
        return {s.name: s for s in self.specs}

    def defaults(self) -> Candidate:
        return Candidate(params={s.name: s.default for s in self.specs})

    def validate(self, cand: Candidate) -> tuple[bool, dict[str, str]]:
        errs: dict[str, str] = {}
        sm = self.spec_map()

        # unknown params
        for k in cand.params:
            if k not in sm:
                errs[k] = "unknown param"

        # validate each spec
        for s in self.specs:
            if s.name not in cand.params:
                errs[s.name] = "missing param"
                continue
            ok, msg = s.validate(cand.params[s.name])
            if not ok:
                errs[s.name] = msg

        # cross-param legality constraints (serving realism)
        if not errs:
            errs.update(self._cross_constraints(cand))

        return (len(errs) == 0), errs

    def _cross_constraints(self, cand: Candidate) -> dict[str, str]:
        """
        Hard constraints that avoid nonsense configs.
        """
        e: dict[str, str] = {}

        # Example constraints
        conc = int(cand.get("concurrency_limit", 1))
        max_new = int(cand.get("max_new_tokens", 256))
        max_model_len = int(cand.get("max_model_len", 4096))
        max_batch_tokens = int(cand.get("max_batch_tokens", 8192))

        if conc <= 0:
            e["concurrency_limit"] = "must be > 0"
        if max_new <= 0:
            e["max_new_tokens"] = "must be > 0"
        if max_model_len < 512:
            e["max_model_len"] = "too small"
        if max_batch_tokens < 256:
            e["max_batch_tokens"] = "too small"

        # batch tokens should be able to hold at least some sequences
        if max_batch_tokens < max_model_len // 4:
            e["max_batch_tokens"] = "max_batch_tokens too small for model_len"

        # kv cache dtype & quantization coherence (simple rules)
        quant = cand.get("quantization", "none")
        kv_dtype = cand.get("kv_cache_dtype", "auto")

        if quant == "fp16" and kv_dtype in ("fp8",):
            e["kv_cache_dtype"] = "fp16 quant with fp8 kv-cache is unusual; disallow for baseline"

        return e

    def sample_random(self, rng: random.Random) -> Candidate:
        params: dict[str, Any] = {}
        for s in self.specs:
            if s.ptype == "int":
                lo = int(s.low) if s.low is not None else 0
                hi = int(s.high) if s.high is not None else lo
                step = int(s.step or 1)
                if hi < lo:
                    lo, hi = hi, lo
                n = ((hi - lo) // step) + 1
                v = lo + step * rng.randrange(max(1, n))
                params[s.name] = int(v)
            elif s.ptype == "float":
                lo = float(s.low) if s.low is not None else 0.0
                hi = float(s.high) if s.high is not None else lo
                if hi < lo:
                    lo, hi = hi, lo
                params[s.name] = float(lo + (hi - lo) * rng.random())
            elif s.ptype == "categorical":
                assert s.choices is not None and len(s.choices) > 0
                params[s.name] = rng.choice(list(s.choices))
            elif s.ptype == "bool":
                params[s.name] = bool(rng.choice([True, False]))
            else:
                params[s.name] = s.default

        cand = Candidate(params=params)
        ok, errs = self.validate(cand)
        if ok:
            return cand

        # repair a bit: fallback to defaults for invalid fields
        repaired = params.copy()
        dflt = self.defaults().params
        for k in errs:
            if k in dflt:
                repaired[k] = dflt[k]
        cand2 = Candidate(params=repaired)
        # if still invalid, just return defaults (rare)
        ok2, _ = self.validate(cand2)
        return cand2 if ok2 else self.defaults()

    def encode_for_pymoo(self, cand: Candidate) -> list[float]:
        """
        Encode candidate to numeric vector for pymoo.
        - int/float: raw
        - bool: 0/1
        - categorical: index in choices
        """
        vec: list[float] = []
        for s in self.specs:
            v = cand.get(s.name, s.default)
            if s.ptype == "int":
                vec.append(float(int(v)))
            elif s.ptype == "float":
                vec.append(float(v))
            elif s.ptype == "bool":
                vec.append(1.0 if bool(v) else 0.0)
            elif s.ptype == "categorical":
                assert s.choices is not None
                vec.append(float(s.choices.index(v)))
            else:
                vec.append(0.0)
        return vec

    def decode_from_pymoo(self, vec: Sequence[float]) -> Candidate:
        """
        Decode numeric vector from pymoo to Candidate with proper types.
        """
        params: dict[str, Any] = {}
        for i, s in enumerate(self.specs):
            x = float(vec[i])
            if s.ptype == "int":
                v = int(round(x))
                # clamp & step-align
                lo = int(s.low) if s.low is not None else v
                hi = int(s.high) if s.high is not None else v
                v = max(lo, min(hi, v))
                if s.step and s.step > 1:
                    base = lo
                    v = base + int(round((v - base) / s.step)) * int(s.step)
                    v = max(lo, min(hi, v))
                params[s.name] = int(v)
            elif s.ptype == "float":
                lo = float(s.low) if s.low is not None else x
                hi = float(s.high) if s.high is not None else x
                params[s.name] = float(max(lo, min(hi, x)))
            elif s.ptype == "bool":
                params[s.name] = bool(x >= 0.5)
            elif s.ptype == "categorical":
                assert s.choices is not None and len(s.choices) > 0
                idx = int(round(x))
                idx = max(0, min(len(s.choices) - 1, idx))
                params[s.name] = s.choices[idx]
            else:
                params[s.name] = s.default

        cand = Candidate(params=params)
        ok, _ = self.validate(cand)
        return cand if ok else self.defaults()

    def to_serving_config(self, cand: Candidate) -> dict[str, Any]:
        """
        Convert candidate to a "serving config" dictionary (backend-agnostic).
        The orchestrator can decide how to apply it (restart server or only report).
        """
        p = cand.to_dict()
        # Split into server-ish and client-ish
        return {
            "client": {
                "concurrency_limit": int(p.get("concurrency_limit", 64)),
                "request_timeout_s": float(p.get("request_timeout_s", 30.0)),
                "max_new_tokens": int(p.get("max_new_tokens", 256)),
            },
            "server": {
                "max_batch_tokens": int(p.get("max_batch_tokens", 8192)),
                "max_num_seqs": int(p.get("max_num_seqs", 64)),
                "max_model_len": int(p.get("max_model_len", 4096)),
                "kv_cache_dtype": p.get("kv_cache_dtype", "auto"),
                "quantization": p.get("quantization", "none"),
                "gpu_memory_utilization": float(p.get("gpu_memory_utilization", 0.90)),
                "enable_prefix_caching": bool(p.get("enable_prefix_caching", True)),
                "tensor_parallel_size": int(p.get("tensor_parallel_size", 1)),
            },
        }


def default_search_space() -> SearchSpace:
    """
    A pragmatic 10-parameter search space (8~12 target).
    Mix of client-side and server-side knobs commonly used in production tuning.
    """
    specs = [
        ParamSpec(
            name="concurrency_limit",
            ptype="int",
            default=64,
            low=8,
            high=512,
            step=8,
            description="Client concurrency cap (also impacts queueing/tail).",
        ),
        ParamSpec(
            name="request_timeout_s",
            ptype="float",
            default=30.0,
            low=5.0,
            high=120.0,
            description="Client timeout (too low increases retry/timeout rate).",
        ),
        ParamSpec(
            name="max_new_tokens",
            ptype="int",
            default=256,
            low=64,
            high=1024,
            step=32,
            description="Generation cap per request.",
        ),
        ParamSpec(
            name="max_batch_tokens",
            ptype="int",
            default=8192,
            low=1024,
            high=65536,
            step=512,
            description="Server batching limit (tokens).",
        ),
        ParamSpec(
            name="max_num_seqs",
            ptype="int",
            default=64,
            low=8,
            high=256,
            step=8,
            description="Server batch sequences cap.",
        ),
        ParamSpec(
            name="max_model_len",
            ptype="int",
            default=4096,
            low=2048,
            high=16384,
            step=256,
            description="Model context length (affects KV memory).",
        ),
        ParamSpec(
            name="kv_cache_dtype",
            ptype="categorical",
            default="auto",
            choices=["auto", "fp16", "bf16", "fp8"],
            description="KV cache dtype (if backend supports).",
        ),
        ParamSpec(
            name="quantization",
            ptype="categorical",
            default="none",
            choices=["none", "int8", "awq", "gptq", "fp16"],
            description="Quantization mode (backend-dependent).",
        ),
        ParamSpec(
            name="gpu_memory_utilization",
            ptype="float",
            default=0.90,
            low=0.50,
            high=0.98,
            description="GPU memory utilization cap (vLLM-like).",
        ),
        ParamSpec(
            name="enable_prefix_caching",
            ptype="bool",
            default=True,
            description="Prefix caching on/off (helps multi-turn / RAG).",
        ),
        ParamSpec(
            name="tensor_parallel_size",
            ptype="int",
            default=1,
            low=1,
            high=8,
            step=1,
            description="TP size (if multi-GPU available).",
        ),
    ]
    # Keep it 10~11 params; adjust by removing TP if you want strict <=10
    return SearchSpace(specs=specs)
