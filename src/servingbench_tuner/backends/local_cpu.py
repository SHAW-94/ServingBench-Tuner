from __future__ import annotations

import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import anyio

from .base import BackendAdapter, GenerationRequest, TokenEvent


@dataclass
class LocalCPUConfig:
    """
    Local CPU backend using transformers.

    Notes:
    - This is optional and best-effort. It's meant for quickstart demos.
    - For real benchmarking, prefer vLLM on GPU.
    """

    model_id: str = "sshleifer/tiny-gpt2"
    device: str = "cpu"
    max_new_tokens_default: int = 128
    temperature_default: float = 0.0


class LocalCPUBackend(BackendAdapter):
    """
    Transformers-based local CPU backend.
    Provides streaming via TextIteratorStreamer when available.

    Requirements:
      pip install -e ".[cpu]"
    """

    name = "local_cpu"

    def __init__(self, cfg: LocalCPUConfig) -> None:
        self.cfg = cfg
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "transformers not installed. Install with: pip install -e '.[cpu]'"
            ) from e

        self._AutoModelForCausalLM = AutoModelForCausalLM
        self._AutoTokenizer = AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(cfg.model_id)
        self.model.to(cfg.device)
        self.model.eval()

    async def health(self) -> bool:
        return True

    def _build_prompt(self, req: GenerationRequest) -> str:
        if req.prompt:
            return req.prompt
        if req.messages:
            # naive chat-to-prompt formatting (simple & deterministic)
            parts = []
            for m in req.messages:
                role = m.get("role", "user")
                content = m.get("content", "")
                parts.append(f"{role}: {content}")
            parts.append("assistant:")
            return "\n".join(parts)
        return ""

    async def generate_stream(self, req: GenerationRequest) -> AsyncIterator[TokenEvent]:
        prompt = self._build_prompt(req)
        max_new = int(req.max_new_tokens or self.cfg.max_new_tokens_default)
        temperature = float(
            req.temperature if req.temperature is not None else self.cfg.temperature_default
        )

        try:
            from transformers import TextIteratorStreamer  # type: ignore
        except Exception:
            # Fallback: no true streaming; do a single generate and return final
            t0 = time.time()
            inputs = self.tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(self.cfg.device) for k, v in inputs.items()}
            with anyio.to_thread.run_sync(lambda: None):
                pass  # keep async contract consistent

            def _run_generate():
                return self.model.generate(
                    **inputs,
                    max_new_tokens=max_new,
                    do_sample=(temperature > 1e-6),
                    temperature=max(temperature, 1e-6),
                )

            out_ids = await anyio.to_thread.run_sync(_run_generate)
            out_text = self.tokenizer.decode(out_ids[0], skip_special_tokens=True)
            # return only the generated suffix as delta (best effort)
            delta = out_text[len(prompt) :] if out_text.startswith(prompt) else out_text
            meta = {"backend": self.name, "ttft_s": time.time() - t0}
            yield TokenEvent(text=delta, is_final=True, usage=None, meta=meta)
            return

        # True streaming path
        t0 = time.time()
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.cfg.device) for k, v in inputs.items()}

        def _generate_blocking():
            # runs in a thread
            self.model.generate(
                **inputs,
                max_new_tokens=max_new,
                do_sample=(temperature > 1e-6),
                temperature=max(temperature, 1e-6),
                streamer=streamer,
            )

        # Launch generation in background thread
        await anyio.to_thread.run_sync(lambda: None)
        tg = anyio.create_task_group()
        meta: dict[str, Any] = {"backend": self.name}
        first = True

        async def _run_gen():
            await anyio.to_thread.run_sync(_generate_blocking)

        async def _stream_tokens():
            nonlocal first, meta
            for text in streamer:
                if first:
                    meta = {**meta, "ttft_s": time.time() - t0}
                    first = False
                yield TokenEvent(text=str(text), is_final=False, usage=None, meta=meta)
            yield TokenEvent(text="", is_final=True, usage=None, meta=meta)

        async with tg:
            tg.start_soon(_run_gen)
            # consume streamer in current task
            async for ev in _stream_tokens():
                yield ev
                if ev.is_final:
                    tg.cancel_scope.cancel()
                    return

    def apply_serving_config(self, cfg: dict[str, Any]) -> None:
        # Local CPU backend doesn't support dynamic reconfiguration beyond request fields
        raise NotImplementedError("LocalCPUBackend does not support apply_serving_config.")
