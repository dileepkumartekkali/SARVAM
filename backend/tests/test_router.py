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


async def test_rate_limited_provider_is_skipped_on_the_next_call_within_a_turn():
    """Live-confirmed real bug: a single turn needing several separate LLM
    calls (draft, tool dispatch, self-check, correction) had EACH call
    independently retry the SAME already-429'd provider from scratch before
    falling through -- wasting a call (and further counting against an
    already-exhausted rate limit) every single time. A provider that just
    returned 429 must be skipped on the NEXT call, not retried blind."""
    primary = FakeProvider("primary", error=LLMProviderError("primary returned 429: rate limited", retriable=True))
    backup = FakeProvider("backup", chunks=["ok"])
    router = LLMRouter([primary, backup])

    await router.complete_with_fallback([{"role": "user", "content": "first call"}])
    assert primary.calls == 1  # first call still had to try it once to learn it's rate-limited
    await router.complete_with_fallback([{"role": "user", "content": "second call, same turn"}])

    assert primary.calls == 1  # NOT retried on the second call -- skipped straight to backup
    assert backup.calls == 2


async def test_a_non_rate_limit_retriable_error_does_not_trigger_cooldown():
    """Only a 429 specifically means "this provider is currently
    exhausted, skip it for a while" -- a different transient error (a
    dropped connection, a 500) might well succeed again on the very next
    call, so it must not be skipped."""
    primary = FakeProvider("primary", error=LLMProviderError("primary returned 503: server error", retriable=True))
    backup = FakeProvider("backup", chunks=["ok"])
    router = LLMRouter([primary, backup])

    await router.complete_with_fallback([{"role": "user", "content": "first call"}])
    await router.complete_with_fallback([{"role": "user", "content": "second call"}])

    assert primary.calls == 2  # retried both times -- no cooldown for a non-429 error
