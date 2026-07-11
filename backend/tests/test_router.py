"""Router fallback logic, tested against fake in-memory providers — no HTTP.

Covers: fallback ordering on retriable errors, fail-fast on non-retriable
errors, and streaming fallback only before the first chunk is yielded.
"""

import pytest

from agent_core.llm_adapter import LLMProviderError, LLMRouter

from ._fakes import FakeProvider


async def _drain(agen):
    return [c async for c in agen]


async def test_complete_falls_through_on_retriable_error():
    primary = FakeProvider("primary", error=LLMProviderError("boom", retriable=True))
    backup = FakeProvider("backup", chunks=["hello"])
    router = LLMRouter([primary, backup])

    result = await router.complete_with_fallback([{"role": "user", "content": "hi"}])

    assert result == "hello"
    assert primary.calls == 1
    assert backup.calls == 1


async def test_complete_fails_fast_on_non_retriable_error():
    primary = FakeProvider("primary", error=LLMProviderError("bad auth", retriable=False))
    backup = FakeProvider("backup", chunks=["hello"])
    router = LLMRouter([primary, backup])

    with pytest.raises(LLMProviderError) as exc_info:
        await router.complete_with_fallback([{"role": "user", "content": "hi"}])

    assert exc_info.value.retriable is False
    assert primary.calls == 1
    assert backup.calls == 0  # never reached — no fallback on non-retriable


async def test_complete_raises_last_error_when_all_providers_fail():
    a = FakeProvider("a", error=LLMProviderError("a down", retriable=True))
    b = FakeProvider("b", error=LLMProviderError("b down", retriable=True))
    router = LLMRouter([a, b])

    with pytest.raises(LLMProviderError, match="b down"):
        await router.complete_with_fallback([{"role": "user", "content": "hi"}])


async def test_stream_falls_through_before_first_chunk():
    primary = FakeProvider("primary", chunks=[], error=LLMProviderError("boom", retriable=True), fail_after=0)
    backup = FakeProvider("backup", chunks=["hel", "lo"])
    router = LLMRouter([primary, backup])

    result = await _drain(router.stream_with_fallback([{"role": "user", "content": "hi"}]))

    assert result == ["hel", "lo"]
    assert primary.calls == 1
    assert backup.calls == 1


async def test_stream_does_not_fall_back_after_first_chunk_even_if_retriable():
    primary = FakeProvider(
        "primary", chunks=["par", "tial"], error=LLMProviderError("dropped", retriable=True), fail_after=1
    )
    backup = FakeProvider("backup", chunks=["should", "not", "run"])
    router = LLMRouter([primary, backup])

    agen = router.stream_with_fallback([{"role": "user", "content": "hi"}])
    with pytest.raises(LLMProviderError, match="dropped"):
        await _drain(agen)

    assert backup.calls == 0  # partial output already emitted — no retroactive fallback
