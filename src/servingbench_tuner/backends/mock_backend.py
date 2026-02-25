from __future__ import annotations

import random
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import anyio

from .base import BackendAdapter, GenerationRequest, TokenEvent


@dataclass
class MockBackendConfig:
    """
    A mock backend that simulates:
    - queueing under concurrency limit
    - prefill time (TTFT-ish)
    - decode time per token (TPOT-ish)
    - deterministic-ish output text stream

    These parameters are intentionally simple but production-relevant:
    you can demonstrate tail latency blowups from queue congestion.
    """

    concurrency: int = 8
    seed: int = 42

    # latency model
    ttft_base_s: float = 0.08  # base TTFT
    prefill_per_input_tok_s: float = 0.00010
    decode_per_output_tok_s: float = 0.010

    # jitter
    jitter_logn_mu: float = 0.0
    jitter_logn_sigma: float = 0.15

    # text output
    token_text: str = "▁"  # a visible token-ish marker


class MockBackend(BackendAdapter):
    """
    Mock backend for CI/CPU:
    - Uses an anyio semaphore to enforce concurrency and produce queue_time
    - Streams tokens with sleeps to emulate TTFT/TPOT
    """

    name = "mock"

    def __init__(self, cfg: MockBackendConfig) -> None:
        self.cfg = cfg
        self._sem = anyio.Semaphore(cfg.concurrency)
        self._rng = random.Random(cfg.seed)

    async def health(self) -> bool:
        return True

    def _jitter(self) -> float:
        # lognormal > 0
        return max(
            0.3, self._rng.lognormvariate(self.cfg.jitter_logn_mu, self.cfg.jitter_logn_sigma)
        )

    def _estimate_times(self, req: GenerationRequest) -> dict[str, float]:
        # Use token hints if provided; otherwise estimate from text length
        in_tok = (
            req.input_tokens
            if req.input_tokens is not None
            else max(16, len(req.prompt or "") // 4)
        )
        out_tok = req.output_tokens if req.output_tokens is not None else int(req.max_new_tokens)

        jitter = self._jitter()
        prefill = (self.cfg.ttft_base_s + self.cfg.prefill_per_input_tok_s * in_tok) * jitter
        decode = (self.cfg.decode_per_output_tok_s * out_tok) * (0.9 + 0.2 * self._rng.random())
        return {
            "prefill_s": prefill,
            "decode_s": decode,
            "in_tok": float(in_tok),
            "out_tok": float(out_tok),
        }

    async def generate_stream(self, req: GenerationRequest) -> AsyncIterator[TokenEvent]:
        t_arrive = time.time()
        meta: dict[str, Any] = {"backend": self.name}

        # queueing
        async with self._sem:
            t_start = time.time()
            queue_time = t_start - t_arrive

            times = self._estimate_times(req)
            prefill_s = float(times["prefill_s"])
            decode_s = float(times["decode_s"])
            out_tok = int(times["out_tok"])

            meta.update(
                {
                    "queue_time_s": queue_time,
                    "prefill_time_s": prefill_s,
                    "decode_time_s": decode_s,
                }
            )

            # prefill sleep (TTFT)
            await anyio.sleep(prefill_s)
            meta["ttft_s"] = time.time() - t_start

            # Emit first token (TTFT boundary)
            if out_tok <= 0:
                yield TokenEvent(text="", is_final=True, usage={"completion_tokens": 0}, meta=meta)
                return

            # stream tokens
            per_tok = decode_s / max(1, out_tok)
            # Use deterministic-ish "token" text; you can switch to " token" for readability
            for _i in range(out_tok):
                await anyio.sleep(per_tok)
                yield TokenEvent(text=self.cfg.token_text, is_final=False, usage=None, meta=meta)

            usage = {
                "prompt_tokens": int(times["in_tok"]),
                "completion_tokens": out_tok,
                "total_tokens": int(times["in_tok"]) + out_tok,
            }
            yield TokenEvent(text="", is_final=True, usage=usage, meta=meta)

    def apply_serving_config(self, cfg: dict[str, Any]) -> None:
        """
        Allow hot update of mock parameters (useful for tuner).
        """
        if "concurrency" in cfg:
            self.cfg.concurrency = int(cfg["concurrency"])
            self._sem = anyio.Semaphore(self.cfg.concurrency)
        if "seed" in cfg:
            self.cfg.seed = int(cfg["seed"])
            self._rng = random.Random(self.cfg.seed)

        # latency knobs
        for k in ["ttft_base_s", "prefill_per_input_tok_s", "decode_per_output_tok_s"]:
            if k in cfg:
                setattr(self.cfg, k, float(cfg[k]))
