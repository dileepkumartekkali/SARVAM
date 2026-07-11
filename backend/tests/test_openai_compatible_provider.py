"""OpenAICompatibleProvider — used for Grok, GPT, and Sarvam LLM alike.

Uses httpx.MockTransport (built into httpx) so no network call and no mocking
library is needed.
"""

import json

import httpx
import pytest

from agent_core.llm_adapter.base import LLMProviderError
from agent_core.llm_adapter.providers.openai_compatible import OpenAICompatibleProvider


def _sse_response(*deltas: str) -> httpx.Response:
    body = "".join(
        f'data: {{"choices":[{{"delta":{{"content":{json.dumps(d)}}}}}]}}\n\n' for d in deltas
    )
    body += "data: [DONE]\n\n"
    return httpx.Response(200, content=body.encode(), headers={"content-type": "text/event-stream"})


async def test_grok_streams_end_to_end(monkeypatch):
    monkeypatch.setenv("GROK_API_KEY", "test-key")
    transport = httpx.MockTransport(lambda request: _sse_response("Hel", "lo"))
    provider = OpenAICompatibleProvider(
        name="grok", base_url="https://api.x.ai/v1", model="grok-4",
        api_key_env="GROK_API_KEY", transport=transport,
    )

    chunks = [c async for c in provider.stream([{"role": "user", "content": "hi"}])]

    assert chunks == ["Hel", "lo"]


async def test_gpt_complete_drains_stream(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    transport = httpx.MockTransport(lambda request: _sse_response("world"))
    provider = OpenAICompatibleProvider(
        name="gpt", base_url="https://api.openai.com/v1", model="gpt-4o",
        api_key_env="OPENAI_API_KEY", transport=transport,
    )

    result = await provider.complete([{"role": "user", "content": "hi"}])

    assert result == "world"


async def test_missing_api_key_is_non_retriable(monkeypatch):
    monkeypatch.delenv("SARVAM_API_KEY", raising=False)
    provider = OpenAICompatibleProvider(
        name="sarvam", base_url="https://api.sarvam.ai/v1", model="sarvam-m",
        api_key_env="SARVAM_API_KEY",
    )

    with pytest.raises(LLMProviderError) as exc_info:
        async for _ in provider.stream([{"role": "user", "content": "hi"}]):
            pass

    assert exc_info.value.retriable is False


async def test_rate_limit_is_retriable(monkeypatch):
    monkeypatch.setenv("GROK_API_KEY", "test-key")
    transport = httpx.MockTransport(lambda request: httpx.Response(429, text="rate limited"))
    provider = OpenAICompatibleProvider(
        name="grok", base_url="https://api.x.ai/v1", model="grok-4",
        api_key_env="GROK_API_KEY", transport=transport,
    )

    with pytest.raises(LLMProviderError) as exc_info:
        async for _ in provider.stream([{"role": "user", "content": "hi"}]):
            pass

    assert exc_info.value.retriable is True


async def test_auth_failure_is_not_retriable(monkeypatch):
    monkeypatch.setenv("GROK_API_KEY", "test-key")
    transport = httpx.MockTransport(lambda request: httpx.Response(401, text="bad key"))
    provider = OpenAICompatibleProvider(
        name="grok", base_url="https://api.x.ai/v1", model="grok-4",
        api_key_env="GROK_API_KEY", transport=transport,
    )

    with pytest.raises(LLMProviderError) as exc_info:
        async for _ in provider.stream([{"role": "user", "content": "hi"}]):
            pass

    assert exc_info.value.retriable is False


async def test_empty_choices_chunk_does_not_crash(monkeypatch):
    """A real bug hit live against Sarvam: it sends a trailing SSE chunk with
    an EMPTY `choices` array (a usage-stats/keep-alive frame) — `choices[0]`
    on it raised an uncaught IndexError that crashed the whole turn."""
    monkeypatch.setenv("SARVAM_API_KEY", "test-key")
    body = (
        'data: {"choices":[{"delta":{"content":"hi"},"finish_reason":null}]}\n\n'
        'data: {"choices":[]}\n\n'
        "data: [DONE]\n\n"
    )
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, content=body.encode(), headers={"content-type": "text/event-stream"})
    )
    provider = OpenAICompatibleProvider(
        name="sarvam", base_url="https://api.sarvam.ai/v1", model="sarvam-30b",
        api_key_env="SARVAM_API_KEY", transport=transport,
    )

    chunks = [c async for c in provider.stream([{"role": "user", "content": "hi"}])]

    assert chunks == ["hi"]


def _reasoning_truncated_sse_response() -> httpx.Response:
    """Real shape hit live against Sarvam: chain-of-thought tokens stream in
    `reasoning_content`, never `content`, and the stream ends with
    `finish_reason: "length"` having never produced a real answer."""
    body = (
        'data: {"choices":[{"delta":{"reasoning_content":"thinking..."},"finish_reason":null}]}\n\n'
        'data: {"choices":[{"delta":{},"finish_reason":"length"}]}\n\n'
        "data: [DONE]\n\n"
    )
    return httpx.Response(200, content=body.encode(), headers={"content-type": "text/event-stream"})


async def test_truncated_reasoning_stream_raises_retriable_error(monkeypatch):
    """A real bug hit live against Sarvam: its models are chain-of-thought
    reasoners that can burn an entire max_tokens budget on internal
    `reasoning_content` and never reach a real answer — a 200 OK with no
    usable content, previously returned to the caller as a silent empty
    string instead of triggering the router's fallback to the next
    provider."""
    monkeypatch.setenv("SARVAM_API_KEY", "test-key")
    transport = httpx.MockTransport(lambda request: _reasoning_truncated_sse_response())
    provider = OpenAICompatibleProvider(
        name="sarvam", base_url="https://api.sarvam.ai/v1", model="sarvam-30b",
        api_key_env="SARVAM_API_KEY", transport=transport,
    )

    with pytest.raises(LLMProviderError) as exc_info:
        async for _ in provider.stream([{"role": "user", "content": "hi"}]):
            pass

    assert exc_info.value.retriable is True
    assert "reasoning" in str(exc_info.value)


async def test_truncated_reasoning_complete_with_tools_raises_retriable_error(monkeypatch):
    monkeypatch.setenv("SARVAM_API_KEY", "test-key")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "length",
                        "message": {"content": None, "reasoning_content": "thinking...", "tool_calls": None},
                    }
                ]
            },
        )

    provider = OpenAICompatibleProvider(
        name="sarvam", base_url="https://api.sarvam.ai/v1", model="sarvam-30b",
        api_key_env="SARVAM_API_KEY", transport=httpx.MockTransport(handler),
    )

    with pytest.raises(LLMProviderError) as exc_info:
        await provider.complete_with_tools([{"role": "user", "content": "hi"}])

    assert exc_info.value.retriable is True


async def test_extra_body_merged_into_request(monkeypatch):
    """Sarvam's `reasoning_effort` has no generic equivalent — passed through
    as an opaque provider-specific field, not hardcoded into the shared
    request-building logic."""
    monkeypatch.setenv("SARVAM_API_KEY", "test-key")
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": "ok"}}]})

    provider = OpenAICompatibleProvider(
        name="sarvam", base_url="https://api.sarvam.ai/v1", model="sarvam-30b",
        api_key_env="SARVAM_API_KEY", transport=httpx.MockTransport(handler),
        extra_body={"reasoning_effort": "low"},
    )

    await provider.complete_with_tools([{"role": "user", "content": "hi"}])

    assert captured["body"]["reasoning_effort"] == "low"
