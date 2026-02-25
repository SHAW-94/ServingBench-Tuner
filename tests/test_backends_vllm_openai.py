from __future__ import annotations

import httpx

from tests.helpers import FakeHTTPClient, FakeResponse, collect_async_iter, run_async


def test_url_prefix_and_health_fallback() -> None:
    from servingbench_tuner.backends.vllm_openai import VLLMOpenAIBackend, VLLMOpenAIConfig

    backend = VLLMOpenAIBackend(VLLMOpenAIConfig(api_prefix="/prefix"))
    fake = FakeHTTPClient(
        get_results=[Exception("boom"), FakeResponse(status_code=204)],
    )
    backend._client = fake

    assert backend._url("/v1/models") == "/prefix/v1/models"
    ok = run_async(backend.health())
    assert ok is True
    assert fake.get_calls == ["/prefix/v1/models", "/prefix/"]

    run_async(backend.aclose())
    assert fake.closed is True


def test_generate_stream_parses_sse_chunks_and_usage() -> None:
    from servingbench_tuner.backends.base import GenerationRequest
    from servingbench_tuner.backends.vllm_openai import VLLMOpenAIBackend, VLLMOpenAIConfig

    backend = VLLMOpenAIBackend(VLLMOpenAIConfig(default_model="m1"))
    lines = [
        'data: {"choices": [{"delta": {"content": "Hi"}}]}',
        "not-json",
        'data: {"choices": [{"delta": {"content": "!"}}], "usage": {"completion_tokens": 2}}',
        "data: [DONE]",
    ]
    fake = FakeHTTPClient(stream_response=FakeResponse(status_code=200, lines=lines))
    backend._client = fake

    req = GenerationRequest(messages=[{"role": "user", "content": "hello"}], timeout_s=1.0)
    events = run_async(collect_async_iter(backend.generate_stream(req)))

    assert [e.text for e in events] == ["Hi", "!", ""]
    assert events[0].is_final is False
    assert events[-1].is_final is True
    assert events[-1].usage == {"completion_tokens": 2}
    assert fake.stream_calls[0]["method"] == "POST"
    assert fake.stream_calls[0]["url"] == "/v1/chat/completions"
    assert fake.stream_calls[0]["json"]["model"] == "m1"

    run_async(backend.aclose())


def test_generate_stream_non_200_returns_final_error_event() -> None:
    from servingbench_tuner.backends.base import GenerationRequest
    from servingbench_tuner.backends.vllm_openai import VLLMOpenAIBackend, VLLMOpenAIConfig

    backend = VLLMOpenAIBackend(VLLMOpenAIConfig(default_model="m1"))
    fake = FakeHTTPClient(
        stream_response=FakeResponse(status_code=500, body=b"server down", lines=[]),
    )
    backend._client = fake

    req = GenerationRequest(messages=[{"role": "user", "content": "hello"}], timeout_s=1.0)
    events = run_async(collect_async_iter(backend.generate_stream(req)))

    assert len(events) == 1
    assert events[0].is_final is True
    assert "vLLM server error 500" in events[0].meta.get("error", "")

    run_async(backend.aclose())


def test_generate_stream_requires_model() -> None:
    from servingbench_tuner.backends.base import GenerationRequest
    from servingbench_tuner.backends.vllm_openai import VLLMOpenAIBackend, VLLMOpenAIConfig

    backend = VLLMOpenAIBackend(VLLMOpenAIConfig(default_model=None))
    backend._client = FakeHTTPClient()

    req = GenerationRequest(messages=[{"role": "user", "content": "hello"}], model=None)

    try:
        run_async(collect_async_iter(backend.generate_stream(req)))
    except ValueError as exc:
        assert "requires req.model or cfg.default_model" in str(exc)
    else:
        raise AssertionError("expected ValueError")

    run_async(backend.aclose())


def test_generate_stream_timeout_maps_to_final_error() -> None:
    from servingbench_tuner.backends.base import GenerationRequest
    from servingbench_tuner.backends.vllm_openai import VLLMOpenAIBackend, VLLMOpenAIConfig

    class TimeoutClient(FakeHTTPClient):
        def stream(self, method: str, url: str, **kwargs):  # type: ignore[override]
            _ = (method, url, kwargs)
            raise httpx.TimeoutException("timeout")

    backend = VLLMOpenAIBackend(VLLMOpenAIConfig(default_model="m1"))
    backend._client = TimeoutClient()

    req = GenerationRequest(messages=[{"role": "user", "content": "hello"}], timeout_s=0.1)
    events = run_async(collect_async_iter(backend.generate_stream(req)))

    assert len(events) == 1
    assert events[0].is_final is True
    assert events[0].meta["error"] == "timeout"

    run_async(backend.aclose())
