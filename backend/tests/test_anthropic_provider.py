"""AnthropicProvider — distinct SSE event schema (content_block_delta)."""

import json

import httpx

from agent_core.llm_adapter.providers.anthropic import AnthropicProvider


def _sse_response(*texts: str) -> httpx.Response:
    body = "".join(
        f'data: {{"type":"content_block_delta","delta":{{"type":"text_delta","text":{json.dumps(t)}}}}}\n\n'
        for t in texts
    )
    body += 'data: {"type":"message_stop"}\n\n'
    return httpx.Response(200, content=body.encode(), headers={"content-type": "text/event-stream"})


async def test_claude_streams_end_to_end(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    transport = httpx.MockTransport(lambda request: _sse_response("Bon", "jour"))
    provider = AnthropicProvider(transport=transport)

    chunks = [c async for c in provider.stream([{"role": "user", "content": "hi"}])]

    assert chunks == ["Bon", "jour"]
