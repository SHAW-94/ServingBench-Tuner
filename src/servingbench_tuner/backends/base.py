from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any


@dataclass
class GenerationRequest:
    """
    A minimal request abstraction shared across backends.

    For chat models (recommended), use `messages`.
    For completion models, you may set `prompt` (and leave messages=None).

    Token lengths (input_tokens/output_tokens) are optional; useful for mock backend simulation.
    """

    model: str | None = None

    # Chat format (OpenAI-compatible)
    messages: list[dict[str, str]] | None = None

    # Completion format (optional)
    prompt: str | None = None

    # Sampling / limits
    max_new_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    stream: bool = True

    # Timeouts / retries handled by caller typically
    timeout_s: float = 60.0

    # Optional hints (used by mock backend)
    input_tokens: int | None = None
    output_tokens: int | None = None

    # Any extra backend-specific fields
    extra: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def simple_chat(user_text: str, **kwargs: Any) -> GenerationRequest:
        return GenerationRequest(messages=[{"role": "user", "content": user_text}], **kwargs)


@dataclass
class TokenEvent:
    """
    Streaming token event.
    - text: newly produced text delta (may be empty for keep-alive)
    - is_final: True when generation finishes
    - usage: optional usage dict (prompt_tokens, completion_tokens, total_tokens)
    - meta: additional metadata (e.g., queue_time_s, ttft_s)
    """

    text: str
    is_final: bool = False
    usage: dict[str, Any] | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    t_s: float = field(default_factory=lambda: time.time())


@dataclass
class GenerationResult:
    """
    Non-streaming result.
    """

    text: str
    usage: dict[str, Any] | None = None
    meta: dict[str, Any] = field(default_factory=dict)


class BackendAdapter(ABC):
    """
    BackendAdapter abstracts an inference backend.
    You can implement streaming (generate_stream) and optionally override generate().

    Minimal contract:
    - health(): backend is reachable/ready
    - generate_stream(): yields TokenEvent deltas
    """

    name: str = "base"

    @abstractmethod
    async def health(self) -> bool: ...

    @abstractmethod
    async def generate_stream(self, req: GenerationRequest) -> AsyncIterator[TokenEvent]: ...

    async def generate(self, req: GenerationRequest) -> GenerationResult:
        """
        Default: consume stream and concatenate.
        """
        chunks: list[str] = []
        last_usage: dict[str, Any] | None = None
        meta: dict[str, Any] = {}
        async for ev in self.generate_stream(req):
            if ev.text:
                chunks.append(ev.text)
            if ev.usage is not None:
                last_usage = ev.usage
            meta.update(ev.meta or {})
            if ev.is_final:
                break
        return GenerationResult(text="".join(chunks), usage=last_usage, meta=meta)

    def apply_serving_config(self, cfg: dict[str, Any]) -> None:
        """
        Optional: apply a serving config (e.g., change engine settings).
        Many backends require a restart; in that case this can raise NotImplementedError.
        """
        raise NotImplementedError("apply_serving_config is not implemented for this backend.")
