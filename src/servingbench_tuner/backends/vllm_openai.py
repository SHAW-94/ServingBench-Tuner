from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import httpx

from .base import BackendAdapter, GenerationRequest, TokenEvent


@dataclass
class VLLMOpenAIConfig:
    """
    Config for vLLM OpenAI-compatible server.

    Example server start:
      vllm serve <model> --host 0.0.0.0 --port 8000 --api-key token-xxx

    base_url example: http://127.0.0.1:8000
    """

    base_url: str = "http://127.0.0.1:8000"
    api_key: str | None = None
    default_model: str | None = None
    request_timeout_s: float = 120.0

    # If your server uses a prefix path, set it here (rare)
    api_prefix: str = ""  # e.g. "/api"


class VLLMOpenAIBackend(BackendAdapter):
    """
    Adapter to a vLLM OpenAI-compatible server using /v1/chat/completions streaming.

    Notes:
    - vLLM streams SSE-like lines: "data: {json}\n\n", ending with "data: [DONE]"
    - We parse delta content from choices[].delta.content
    """

    name = "vllm_openai"

    def __init__(self, cfg: VLLMOpenAIConfig) -> None:
        self.cfg = cfg
        headers = {}
        if cfg.api_key:
            headers["Authorization"] = f"Bearer {cfg.api_key}"
        self._client = httpx.AsyncClient(
            base_url=cfg.base_url.rstrip("/"),
            headers=headers,
            timeout=httpx.Timeout(cfg.request_timeout_s),
        )

    def _url(self, path: str) -> str:
        prefix = self.cfg.api_prefix.rstrip("/")
        if prefix:
            return f"{prefix}{path}"
        return path

    async def aclose(self) -> None:
        await self._client.aclose()

    async def health(self) -> bool:
        """
        Try /v1/models. If fails, try GET /.
        """
        try:
            r = await self._client.get(self._url("/v1/models"))
            if r.status_code == 200:
                return True
        except Exception:
            pass
        try:
            r = await self._client.get(self._url("/"))
            return r.status_code < 500
        except Exception:
            return False

    async def generate_stream(self, req: GenerationRequest) -> AsyncIterator[TokenEvent]:
        model = req.model or self.cfg.default_model
        if not model:
            raise ValueError("VLLMOpenAIBackend requires req.model or cfg.default_model")

        if not req.messages and not req.prompt:
            raise ValueError(
                "GenerationRequest must include messages (chat) or prompt (completion)"
            )

        # Prefer chat completion
        messages: list[dict[str, str]]
        messages = req.messages if req.messages else [{"role": "user", "content": req.prompt or ""}]

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
            "temperature": req.temperature,
            "top_p": req.top_p,
            # OpenAI uses max_tokens; we map from max_new_tokens
            "max_tokens": int(req.max_new_tokens),
        }
        payload.update(req.extra or {})

        t0 = time.time()
        meta: dict[str, Any] = {"backend": self.name}

        # Streaming request
        try:
            async with self._client.stream(
                "POST",
                self._url("/v1/chat/completions"),
                json=payload,
                timeout=httpx.Timeout(req.timeout_s),
            ) as resp:
                if resp.status_code != 200:
                    text = await resp.aread()
                    raise RuntimeError(
                        f"vLLM server error {resp.status_code}: {text.decode('utf-8', errors='ignore')}"
                    )

                first_token_emitted = False
                usage: dict[str, Any] | None = None

                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    # vLLM uses "data: ..."
                    data = (
                        line[len("data:") :].strip() if line.startswith("data:") else line.strip()
                    )

                    if not data:
                        continue
                    if data == "[DONE]":
                        # final
                        yield TokenEvent(text="", is_final=True, usage=usage, meta=meta)
                        return

                    # parse json chunk
                    try:
                        obj = json.loads(data)
                    except json.JSONDecodeError:
                        # ignore non-json lines
                        continue

                    # usage may appear at end or not at all depending on server
                    if "usage" in obj and isinstance(obj["usage"], dict):
                        usage = obj["usage"]

                    # Extract delta content (OpenAI chat streaming format)
                    delta_text = ""
                    try:
                        choices = obj.get("choices", [])
                        if choices and isinstance(choices, list):
                            delta = choices[0].get("delta", {})
                            if isinstance(delta, dict):
                                delta_text = delta.get("content") or ""
                    except Exception:
                        delta_text = ""

                    if delta_text:
                        if not first_token_emitted:
                            meta = {**meta, "ttft_s": time.time() - t0}
                            first_token_emitted = True
                        yield TokenEvent(text=delta_text, is_final=False, usage=usage, meta=meta)

        except httpx.TimeoutException:
            yield TokenEvent(text="", is_final=True, usage=None, meta={**meta, "error": "timeout"})
        except Exception as e:
            yield TokenEvent(text="", is_final=True, usage=None, meta={**meta, "error": str(e)})

    def apply_serving_config(self, cfg: dict[str, Any]) -> None:
        """
        vLLM server parameters are usually applied on process start.
        For now, we don't attempt hot-reload from this adapter.
        """
        raise NotImplementedError(
            "vLLM serving config usually requires restarting the server. "
            "Use scripts/start_vllm_server.sh with a config YAML."
        )
