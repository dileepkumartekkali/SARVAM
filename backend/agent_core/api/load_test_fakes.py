"""Fake LLM provider used ONLY when MAAV_LOAD_TEST_MODE=true — see
speech_gateway/load_test_fakes.py's docstring for why this exists: load
testing needs a way to measure the backend's own request-handling capacity
without depending on a real (rate-limited, billed) LLM provider.
"""

from __future__ import annotations

import asyncio


class LoadTestFakeProvider:
    name = "load-test-fake-llm"

    async def complete(self, messages, *, system=None, max_tokens=None, temperature=None):
        await asyncio.sleep(0.02)  # simulate minimal realistic LLM latency
        return "This is a simulated load-test response."

    async def stream(self, messages, *, system=None, max_tokens=None, temperature=None):
        for chunk in ("This ", "is ", "a ", "simulated ", "response."):
            await asyncio.sleep(0.005)
            yield chunk
