from __future__ import annotations

import sys
from collections.abc import AsyncIterator, Iterable
from pathlib import Path
from typing import Any

import anyio

# Ensure src/ imports work whether package is installed or not.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))


def run_async(coro: Any) -> Any:
    """Run a coroutine in tests without requiring pytest-anyio."""

    async def _runner() -> Any:
        return await coro

    return anyio.run(_runner)


async def collect_async_iter(ait: AsyncIterator[Any]) -> list[Any]:
    out: list[Any] = []
    async for item in ait:
        out.append(item)
    return out


def mk_e2e(**overrides: Any) -> Any:
    from servingbench_tuner.core.types import E2EMetrics

    base: dict[str, Any] = {
        "ttft_p50": 0.10,
        "ttft_p95": 0.20,
        "tpot_p50": 0.01,
        "tpot_p95": 0.02,
        "latency_p50": 0.30,
        "latency_p95": 0.60,
        "latency_p99": 0.90,
        "rps": 10.0,
        "tok_s": 100.0,
        "timeout_rate": 0.0,
        "error_rate": 0.0,
        "retry_rate": 0.0,
        "jitter_std": 0.05,
        "tail_amp": 1.50,
        "queue_p95": 0.01,
        "prefill_p95": 0.10,
        "decode_p95": 0.20,
    }
    base.update(overrides)
    return E2EMetrics(**base)


def mk_quality(overall: float = 1.0, **overrides: Any) -> Any:
    from servingbench_tuner.core.types import QualitySummary

    base: dict[str, Any] = {
        "overall": overall,
        "pass_rate": overall,
        "by_type": {"default": overall},
        "details_path": "details.json",
    }
    base.update(overrides)
    return QualitySummary(**base)


def mk_arrival_event(
    request_id: str = "req-1",
    *,
    ts: float = 0.0,
    category: str = "unit",
    input_tokens: int | None = 8,
    output_tokens: int | None = 4,
) -> Any:
    from servingbench_tuner.workload.replay import ArrivalEvent

    return ArrivalEvent(
        timestamp_s=ts,
        request_id=request_id,
        category=category,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


class FakeResponse:
    def __init__(
        self, status_code: int = 200, body: bytes = b"", lines: Iterable[str] | None = None
    ) -> None:
        self.status_code = status_code
        self._body = body
        self._lines = list(lines or [])

    async def aread(self) -> bytes:
        return self._body

    async def aiter_lines(self) -> AsyncIterator[str]:
        for line in self._lines:
            yield line

    async def __aenter__(self) -> FakeResponse:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False


class FakeHTTPClient:
    def __init__(
        self,
        *,
        get_results: Iterable[Any] | None = None,
        stream_response: FakeResponse | None = None,
    ) -> None:
        self._get_results = list(get_results or [])
        self._stream_response = stream_response or FakeResponse()
        self.get_calls: list[str] = []
        self.stream_calls: list[dict[str, Any]] = []
        self.closed = False

    async def get(self, url: str) -> Any:
        self.get_calls.append(url)
        if not self._get_results:
            return FakeResponse(status_code=200)
        item = self._get_results.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def stream(self, method: str, url: str, **kwargs: Any) -> FakeResponse:
        self.stream_calls.append({"method": method, "url": url, **kwargs})
        return self._stream_response

    async def aclose(self) -> None:
        self.closed = True
