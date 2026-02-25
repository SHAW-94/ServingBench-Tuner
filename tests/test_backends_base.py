from __future__ import annotations

from tests.helpers import run_async


def test_generation_request_simple_chat() -> None:
    from servingbench_tuner.backends.base import GenerationRequest

    req = GenerationRequest.simple_chat("hello", max_new_tokens=7)
    assert req.messages == [{"role": "user", "content": "hello"}]
    assert req.max_new_tokens == 7


def test_backend_generate_concatenates_stream_and_usage() -> None:
    from servingbench_tuner.backends.base import BackendAdapter, GenerationRequest, TokenEvent

    class DummyBackend(BackendAdapter):
        name = "dummy"

        async def health(self) -> bool:
            return True

        async def generate_stream(self, req: GenerationRequest):
            _ = req
            yield TokenEvent(text="he", meta={"a": 1})
            yield TokenEvent(text="llo", usage={"completion_tokens": 1}, meta={"b": 2})
            yield TokenEvent(text="", is_final=True, usage={"completion_tokens": 2}, meta={"c": 3})

    backend = DummyBackend()
    req = GenerationRequest.simple_chat("x")
    result = run_async(backend.generate(req))

    assert result.text == "hello"
    assert result.usage == {"completion_tokens": 2}
    assert result.meta == {"a": 1, "b": 2, "c": 3}
