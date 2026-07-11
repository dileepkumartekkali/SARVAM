"""Minimal SSE line parser shared by every provider adapter.

Grok, GPT, Sarvam, Claude, and Gemini all frame their streaming responses as
standard Server-Sent Events (`data: <payload>` lines, blank line between
events) — only the JSON *inside* `data:` differs per provider. One parser,
reused by every adapter, instead of five near-identical ones.
"""

from __future__ import annotations

from typing import AsyncIterator

import httpx


async def iter_sse_data(response: httpx.Response) -> AsyncIterator[str]:
    """Yield each event's raw `data:` payload; stop at a `[DONE]` sentinel."""
    async for line in response.aiter_lines():
        if not line or not line.startswith("data:"):
            continue
        payload = line[len("data:") :].strip()
        if payload == "[DONE]":
            return
        yield payload
