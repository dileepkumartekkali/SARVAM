"""GeminiProvider — distinct request/response schema (contents/candidates)."""

import json

import httpx
import pytest

from agent_core.llm_adapter.base import LLMProviderError
from agent_core.llm_adapter.providers.gemini import GeminiProvider


def _sse_response(*texts: str) -> httpx.Response:
    body = "".join(
        f'data: {{"candidates":[{{"content":{{"parts":[{{"text":{json.dumps(t)}}}]}}}}]}}\n\n'
        for t in texts
    )
    return httpx.Response(200, content=body.encode(), headers={"content-type": "text/event-stream"})


async def test_gemini_streams_end_to_end(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    transport = httpx.MockTransport(lambda request: _sse_response("Nam", "aste"))
    provider = GeminiProvider(transport=transport)

    chunks = [c async for c in provider.stream([{"role": "user", "content": "hi"}])]

    assert chunks == ["Nam", "aste"]


async def test_deprecated_model_404_is_retriable_not_a_hard_crash(monkeypatch):
    """A real bug hit live: `gemini-2.5-flash` (the configured default) 404s
    with "no longer available to new users" against a real key. 404 is
    non-retriable everywhere else (`_http_common.status_retriable`) — but
    here it's a provider-availability fact, not a caller bug. Left
    non-retriable, this would hard-crash every single turn instead of
    falling back to the next configured provider."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            404, json={"error": {"message": "This model is no longer available to new users."}}
        )
    )
    provider = GeminiProvider(transport=transport)

    with pytest.raises(LLMProviderError) as exc_info:
        async for _ in provider.stream([{"role": "user", "content": "hi"}]):
            pass

    assert exc_info.value.retriable is True


async def test_quota_exceeded_429_is_retriable(monkeypatch):
    """Also hit live: a 429 with zero free-tier quota — already retriable via
    the shared status_retriable() path, confirmed still true here."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    transport = httpx.MockTransport(
        lambda request: httpx.Response(429, json={"error": {"message": "quota exceeded, limit: 0"}})
    )
    provider = GeminiProvider(transport=transport)

    with pytest.raises(LLMProviderError) as exc_info:
        await provider.complete_with_tools([{"role": "user", "content": "hi"}])

    assert exc_info.value.retriable is True
