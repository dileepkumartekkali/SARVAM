from agent_core.speech_gateway.failure_policy import (
    is_low_confidence_transcript,
    stt_stream_with_backoff_then_rest,
    tts_synthesize_with_retry_then_fallback,
)


def test_low_confidence_transcript_flagged():
    event = {"type": "transcript", "is_final": True, "text": "uh maybe", "confidence": 0.2}
    assert is_low_confidence_transcript(event) is True


def test_empty_final_transcript_flagged():
    event = {"type": "transcript", "is_final": True, "text": "", "confidence": 0.9}
    assert is_low_confidence_transcript(event) is True


def test_high_confidence_transcript_not_flagged():
    event = {"type": "transcript", "is_final": True, "text": "book a flight", "confidence": 0.95}
    assert is_low_confidence_transcript(event) is False


def test_partial_transcript_never_flagged():
    event = {"type": "transcript", "is_final": False, "text": "", "confidence": 0.1}
    assert is_low_confidence_transcript(event) is False


async def test_stt_backoff_falls_back_to_rest_within_bounded_time():
    attempts = []

    async def always_fails():
        attempts.append(1)
        raise RuntimeError("dropped")
        yield  # pragma: no cover

    async def rest_fallback():
        return {"text": "recovered"}

    events = [
        e
        async for e in stt_stream_with_backoff_then_rest(always_fails, rest_fallback, max_backoff_seconds=0.3)
    ]

    assert events[-1]["via"] == "rest_fallback"
    assert events[-1]["text"] == "recovered"
    assert len(attempts) >= 2  # retried at least once before falling back


async def test_stt_backoff_calls_on_retry_hook_for_each_retry():
    retry_count = 0

    async def always_fails():
        raise RuntimeError("dropped")
        yield  # pragma: no cover

    async def rest_fallback():
        return {"text": "recovered"}

    def on_retry():
        nonlocal retry_count
        retry_count += 1

    _ = [
        e
        async for e in stt_stream_with_backoff_then_rest(
            always_fails, rest_fallback, max_backoff_seconds=0.3, on_retry=on_retry
        )
    ]

    assert retry_count >= 1


async def test_stt_backoff_succeeds_on_a_later_attempt_without_falling_back():
    call_count = 0

    async def fails_once_then_succeeds():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("dropped")
            yield  # pragma: no cover
        yield {"type": "transcript", "text": "hi", "is_final": True}

    async def rest_fallback():
        raise AssertionError("must not be called — the retry succeeded")

    events = [
        e async for e in stt_stream_with_backoff_then_rest(fails_once_then_succeeds, rest_fallback, max_backoff_seconds=2.0)
    ]

    assert events == [{"type": "transcript", "text": "hi", "is_final": True}]


async def test_tts_retries_once_then_falls_back_to_text_only():
    attempts = []
    fallback_called = False

    async def always_fails():
        attempts.append(1)
        raise RuntimeError("tts socket failed")
        yield  # pragma: no cover

    async def on_fallback():
        nonlocal fallback_called
        fallback_called = True

    chunks = [c async for c in tts_synthesize_with_retry_then_fallback(always_fails, on_text_only_fallback=on_fallback)]

    assert chunks == []
    assert len(attempts) == 2  # one retry, on a fresh call each time
    assert fallback_called is True


async def test_tts_succeeds_on_retry_without_falling_back():
    call_count = 0

    async def fails_once_then_succeeds():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("dropped")
            yield  # pragma: no cover
        yield b"audio bytes"

    async def on_fallback():
        raise AssertionError("must not be called — the retry succeeded")

    chunks = [
        c
        async for c in tts_synthesize_with_retry_then_fallback(
            fails_once_then_succeeds, on_text_only_fallback=on_fallback
        )
    ]

    assert chunks == [b"audio bytes"]
