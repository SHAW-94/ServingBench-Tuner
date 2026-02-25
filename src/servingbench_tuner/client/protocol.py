from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import httpx

from servingbench_tuner.backends.base import BackendAdapter, GenerationRequest, TokenEvent


@dataclass
class OpenAIClientConfig:
    """
    OpenAI-compatible streaming client config.

    Works with:
      - vLLM OpenAI server (/v1/chat/completions) streaming
      - Most OpenAI-style servers that use SSE lines: "data: {...}" and "[DONE]"
    """

    base_url: str = "http://127.0.0.1:8000"
    api_key: str | None = None
    default_model: str | None = None
    api_prefix: str = ""  # rarely needed
    connect_timeout_s: float = 10.0
    read_timeout_s: float = 120.0


def _sse_extract_data(line: str) -> str | None:
    line = line.strip()
    if not line:
        return None
    # typical: "data: {...}"
    if line.startswith("data:"):
        return line[len("data:") :].strip()
    # sometimes servers may send raw json lines
    return line


def _chat_payload(req: GenerationRequest, model: str) -> dict[str, Any]:
    if not req.messages and not req.prompt:
        raise ValueError("GenerationRequest must include messages (chat) or prompt (completion)")
    messages = req.messages or [{"role": "user", "content": req.prompt or ""}]

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": True,
        "temperature": float(req.temperature),
        "top_p": float(req.top_p),
        # OpenAI uses max_tokens; map from max_new_tokens
        "max_tokens": int(req.max_new_tokens),
    }
    payload.update(req.extra or {})
    return payload


def _parse_delta_text(obj: dict[str, Any]) -> str:
    """
    OpenAI chat streaming format:
      { choices: [ { delta: { content: "..." } } ] }
    """
    try:
        choices = obj.get("choices", [])
        if not choices:
            return ""
        delta = choices[0].get("delta", {}) or {}
        if isinstance(delta, dict):
            return delta.get("content") or ""
    except Exception:
        return ""
    return ""


class OpenAIStreamingClient(BackendAdapter):
    """
    Treat OpenAI-compatible HTTP server as a BackendAdapter so the same Runner works
    for mock/local_cpu/vllm_openai/openai-compatible server.

    generate_stream yields TokenEvent with:
      - meta.ttft_s set when first token arrives
      - usage if present in stream chunks
    """

    name = "openai_http"

    def __init__(self, cfg: OpenAIClientConfig) -> None:
        self.cfg = cfg
        headers = {}
        if cfg.api_key:
            headers["Authorization"] = f"Bearer {cfg.api_key}"

        self._client = httpx.AsyncClient(
            base_url=cfg.base_url.rstrip("/"),
            headers=headers,
            timeout=httpx.Timeout(
                connect=cfg.connect_timeout_s,
                read=cfg.read_timeout_s,
                write=cfg.read_timeout_s,
                pool=cfg.read_timeout_s,
            ),
        )

    def _url(self, path: str) -> str:
        prefix = self.cfg.api_prefix.rstrip("/")
        if prefix:
            return f"{prefix}{path}"
        return path

    async def aclose(self) -> None:
        await self._client.aclose()

    async def health(self) -> bool:
        # Try /v1/models, fallback /
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
            raise ValueError("OpenAIStreamingClient requires req.model or cfg.default_model")

        payload = _chat_payload(req, model=model)

        meta: dict[str, Any] = {"backend": self.name, "base_url": self.cfg.base_url}
        t_send = time.time()

        first_token = False
        usage: dict[str, Any] | None = None

        try:
            async with self._client.stream(
                "POST",
                self._url("/v1/chat/completions"),
                json=payload,
                timeout=httpx.Timeout(req.timeout_s),
            ) as resp:
                if resp.status_code != 200:
                    body = await resp.aread()
                    raise RuntimeError(
                        f"OpenAI-compatible server error {resp.status_code}: "
                        f"{body.decode('utf-8', errors='ignore')}"
                    )

                async for line in resp.aiter_lines():
                    data = _sse_extract_data(line)
                    if data is None:
                        continue

                    if data == "[DONE]":
                        yield TokenEvent(text="", is_final=True, usage=usage, meta=meta)
                        return

                    try:
                        obj = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    if isinstance(obj, dict) and "usage" in obj and isinstance(obj["usage"], dict):
                        usage = obj["usage"]

                    delta = _parse_delta_text(obj) if isinstance(obj, dict) else ""
                    if delta:
                        if not first_token:
                            meta = {**meta, "ttft_s": time.time() - t_send}
                            first_token = True
                        yield TokenEvent(text=delta, is_final=False, usage=usage, meta=meta)

        except httpx.TimeoutException:
            yield TokenEvent(text="", is_final=True, usage=None, meta={**meta, "error": "timeout"})
        except Exception as e:
            yield TokenEvent(text="", is_final=True, usage=None, meta={**meta, "error": str(e)})

    def apply_serving_config(self, cfg: dict[str, Any]) -> None:
        raise NotImplementedError(
            "OpenAI-compatible HTTP servers typically require restart to apply serving config. "
            "Use your server launcher (e.g., scripts/start_vllm_server.sh)."
        )
