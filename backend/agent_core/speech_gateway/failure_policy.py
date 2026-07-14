"""Failure-matrix policy (S2S plan §5) for the full-duplex gateway — small,
independently-testable pieces rather than one monolithic handler, since each
row of the matrix is a distinct decision:

  STT drop mid-utterance      -> reconnect w/ backoff; REST fallback if the
                                  backoff would exceed ~2s
  low-confidence transcript   -> spoken clarifying question, never a silent
                                  retry loop
  TTS failure                 -> retry once on a fresh socket, then a fixed
                                  text-only + pre-recorded-clip fallback
  LLM provider timeout        -> a fixed "let me get back to you" line —
                                  never dead air
  reconnect mid-session       -> resumed via the LangGraph checkpointer's
                                  thread_id (already true since Phase 2/3;
                                  nothing new needed here)
  malformed audio              -> rejected pre-Sarvam (agent_core.speech.
                                  audio_validation; already enforced in Phase 4)
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator, Awaitable, Callable

from ..agents.task_agent import CLARIFYING_QUESTION, LLM_UNAVAILABLE_APOLOGY

TRANSCRIPT_CONFIDENCE_THRESHOLD = 0.4

# Reused rather than duplicated — task_agent.py had its own independently
# hand-written copy of this same line until this was caught in a pre-deploy
# sweep; two apologies for the same failure meant they could silently drift.
LLM_TIMEOUT_APOLOGY = LLM_UNAVAILABLE_APOLOGY
TTS_FAILURE_APOLOGY = "I'm having trouble with audio right now."

# Reused rather than duplicated — a low-confidence transcript and a
# low-confidence language both resolve to "ask, don't guess."
LOW_CONFIDENCE_TRANSCRIPT_QUESTION = CLARIFYING_QUESTION


def is_low_confidence_transcript(event: dict) -> bool:
    """True for a final transcript whose confidence is below threshold, or an
    empty final transcript — both must trigger a clarifying question, never a
    silent retry."""
    if event.get("type") != "transcript" or not event.get("is_final"):
        return False
    text = (event.get("text") or "").strip()
    confidence = event.get("confidence")
    if not text:
        return True
    return confidence is not None and confidence < TRANSCRIPT_CONFIDENCE_THRESHOLD


async def stt_stream_with_backoff_then_rest(
    stream_attempt: Callable[[], AsyncIterator[dict]],
    rest_fallback: Callable[[], Awaitable[dict]],
    *,
    max_backoff_seconds: float = 2.0,
    on_retry: Callable[[], None] | None = None,
) -> AsyncIterator[dict]:
    """Retries the whole STT stream attempt with exponential backoff; once the
    next backoff would push total waiting past `max_backoff_seconds`, gives up
    and falls back to REST on whatever audio the caller buffered — never
    retries forever and never silently drops the utterance.

    `stream_attempt` is called fresh on every retry — pass a closure over the
    SAME underlying audio source (e.g. a generator already reading off the
    client's WebSocket) so a retry continues that live stream into a new
    Sarvam connection rather than trying to replay already-sent frames.
    """
    delay = 0.25
    elapsed = 0.0
    last_error: Exception | None = None

    while elapsed < max_backoff_seconds:
        try:
            async for event in stream_attempt():
                yield event
            return
        except Exception as e:  # noqa: BLE001 — any stream failure triggers the same policy
            last_error = e
            if elapsed + delay >= max_backoff_seconds:
                break
            if on_retry:
                on_retry()
            await asyncio.sleep(delay)
            elapsed += delay
            delay = min(delay * 2, max_backoff_seconds - elapsed)

    try:
        result = await rest_fallback()
        yield {**result, "via": "rest_fallback", "after_error": str(last_error)}
    except Exception as rest_error:
        yield {"type": "error", "reason": f"{last_error}; REST fallback also failed: {rest_error}"}


async def tts_synthesize_with_retry_then_fallback(
    primary_attempt: Callable[[], AsyncIterator[bytes]],
    *,
    on_text_only_fallback: Callable[[], Awaitable[None]],
) -> AsyncIterator[bytes]:
    """Retries the TTS synthesis once on a fresh socket (a new call to
    `primary_attempt`, which opens its own connection); on a second failure,
    falls back to a text-only signal — via `on_text_only_fallback` — instead
    of silently producing no audio. Callers should play a pre-recorded "having
    trouble with audio" clip alongside the text-only fallback; picking and
    shipping that actual audio asset is outside this module's scope.
    """
    for attempt in range(2):
        try:
            async for chunk in primary_attempt():
                yield chunk
            return
        except Exception:
            if attempt == 1:
                await on_text_only_fallback()
                return
            continue
