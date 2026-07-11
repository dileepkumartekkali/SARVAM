"""Sarvam STT/TTS client contracts — interface only, no implementations.

These mirror Sarvam's *actual* three transports (S2S plan §1), which are three
genuinely different tools, not one API with a flag:

  STT  streaming  wss://…/speech-to-text/ws — live mic, PCM16/WAV only
       batch      async, uploads up to ~2h
       rest       sync, files < 30s
  TTS  streaming  WebSocket, one socket per utterance, NO cancel primitive

Constraints baked into the signatures (from docs/maav memory — easy to violate):

- STT streaming accepts **PCM16/WAV only** (MP3/AAC/OGG are batch/REST-only).
  `codec` is required on the streaming call so no caller forgets.
- STT streaming yields VAD signals (`speech_start`/`speech_end`) alongside
  transcripts — barge-in and end-of-turn depend on them, so they are part of
  the stream contract, not a side channel.
- The STT language guess is not trusted: transcripts carry the raw hint, and a
  separate language agent scores confidence. Nothing here promises a language.
- TTS has **no cancel**. A stream is single-utterance-lifetime; interruption is
  "stop consuming + close the socket + open a new one" at the gateway, so there
  is deliberately no `cancel()` method to imply otherwise.
"""

from __future__ import annotations

from enum import Enum
from typing import AsyncIterator, Protocol, runtime_checkable


class STTMode(str, Enum):
    """Sarvam Saaras streaming modes (S2S plan §1)."""

    TRANSCRIBE = "transcribe"
    TRANSLATE = "translate"
    VERBATIM = "verbatim"
    TRANSLIT = "translit"
    CODEMIX = "codemix"  # first-class code-mix mode — use directly, don't hand-roll


# A transcript/VAD event off the STT stream. Kept as a plain dict at this layer
# (e.g. {"type": "vad", "signal": "speech_start"} or
# {"type": "transcript", "text": ..., "is_final": ..., "language_hint": ...}).
# Promote to a pydantic model in Phase 3 once the field set is exercised.
STTEvent = dict[str, object]


@runtime_checkable
class SpeechSTTClient(Protocol):
    """Sarvam speech-to-text across all three transports."""

    def stream(
        self,
        audio: AsyncIterator[bytes],
        *,
        codec: str,               # pcm_s16le | pcm_l16 | pcm_raw | wav — required
        sample_rate: int = 16000,
        mode: STTMode = STTMode.CODEMIX,
        vad_signals: bool = True,
        high_vad_sensitivity: bool = True,
    ) -> AsyncIterator[STTEvent]:
        """Live transcription over the Sarvam STT WebSocket.

        Yields interleaved VAD signals and (partial/final) transcripts. Must
        reconnect with exponential backoff on WS drop (drops are normal) and,
        past a bounded retry window, the gateway falls back to `transcribe_rest`
        on buffered audio rather than losing the utterance.
        """
        ...

    async def transcribe_rest(self, audio: bytes, *, mode: STTMode = STTMode.CODEMIX) -> STTEvent:
        """Sync REST transcription for short clips (< 30s)."""
        ...

    async def transcribe_batch(self, audio_uri: str, *, mode: STTMode = STTMode.CODEMIX) -> str:
        """Async batch transcription for long uploads (up to ~2h). Returns a job id."""
        ...


@runtime_checkable
class SpeechTTSClient(Protocol):
    """Sarvam text-to-speech. No cancel primitive — see module docstring."""

    def synthesize(
        self,
        text_chunks: AsyncIterator[str],
        *,
        language: str,
        model: str = "bulbul:v3",   # v2 uses pitch/loudness; v3 uses temperature — model-aware config
        voice: str | None = None,
        pace: float | None = None,
    ) -> AsyncIterator[bytes]:
        """Open one TTS socket for a single utterance and stream audio chunks back.

        Reuses the socket for sequential `convert()` calls within one reply, then
        the socket dies with the utterance. To interrupt: stop consuming this
        iterator and close it — there is no in-band cancel.
        """
        ...
