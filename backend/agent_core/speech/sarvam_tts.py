"""Sarvam TTS WebSocket adapter — bulbul:v2/v3, model-aware params, one socket
per utterance (S2S plan §2-3: opened once, reused for sequential `text`
messages, dies with the utterance — never opened per chunk).

Message protocol confirmed against Sarvam's live docs
(docs.sarvam.ai/api-reference-docs/text-to-speech/stream.md, fetched July
2026) — this replaced an earlier best-guess protocol (flat fields, a
`convert` message type, raw binary audio frames) that was **proven wrong** by
a real connection attempt: Sarvam returned `422 Input parameters has to be a
valid dictionary`, because the real API nests everything under a `data`
object and uses `target_language_code`/`speaker`, not `language`/`voice`.
Audio comes back as base64 inside a JSON `audio` message, not a raw binary
WS frame.

Model-aware params: bulbul:v2 takes pitch/loudness (pace 0.3-3.0); bulbul:v3
takes temperature instead (pace 0.5-2.0, preprocessing always on) — sending
v2 params to v3 is rejected, so this adapter only ever sends the params that
match the selected model.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
from typing import AsyncIterator, Callable

import websockets
import websockets.exceptions

from ..security.pii import mask_pii

logger = logging.getLogger("agent_core.speech")

# How long to wait, after the last audio chunk for a flushed text message,
# before deciding the server is done with it. Real-API testing found Sarvam
# does not reliably send the documented `{"type":"event","data":
# {"event_type":"final"}}` completion marker (tried with and without the
# `send_completion_event` config flag — neither produced one). If a "final"
# event genuinely does arrive it's still honored immediately; this timeout
# is the fallback for when it doesn't, not the primary signal.
_CHUNK_IDLE_TIMEOUT_SECONDS = 2.0

_DEFAULT_WS_URL = "wss://api.sarvam.ai/text-to-speech/ws"

_PACE_RANGES = {"bulbul:v2": (0.3, 3.0), "bulbul:v3": (0.5, 2.0)}

# Real bug hit live: reported as "TTS speaking not clear" -- Sarvam's own
# docs (docs.sarvam.ai/api-reference/text-to-speech/stream, confirmed live)
# say `output_audio_codec` defaults to MP3, not raw PCM, when this config
# field is omitted -- which it always was here. Each streamed "audio" event
# is a FRAGMENT of a continuous MP3 stream, not a self-contained file; MP3
# decodes chunk-by-chunk (this app's whole playback model, ttsPlayback.js)
# have frame-to-frame bit-reservoir dependencies, so decoding arbitrary
# fragments in isolation produces exactly the glitchy/unclear audio
# reported. Requesting uncompressed linear16 PCM instead means every chunk
# decodes cleanly and independently, matching how it's actually played.
_OUTPUT_AUDIO_CODEC = "linear16"
# 24000 Hz -- confirmed live to actually work with this account's real
# bulbul:v2 usage (a real synthesize() call returned real audio bytes at
# this rate), not the 22050 Hz frontend/src/api/ttsPlayback.js's fallback
# PCM decoder used to guess (never verified against a real payload).
# Requested explicitly so both sides agree on a known rate instead of each
# independently guessing.
_SPEECH_SAMPLE_RATE = 24000

# Sarvam's TTS language codes are "<lang>-IN", not the bare 2-letter codes
# language_agent uses elsewhere in this codebase.
_LANGUAGE_TO_SARVAM_CODE = {
    "hi": "hi-IN", "bn": "bn-IN", "ta": "ta-IN", "te": "te-IN", "gu": "gu-IN",
    "kn": "kn-IN", "ml": "ml-IN", "mr": "mr-IN", "pa": "pa-IN", "or": "od-IN",
    "en": "en-IN",
}

# A real, live-confirmed bug briefly lived here: an earlier key on this
# account had no bulbul:v3 access at all -- every v3-only speaker
# ("shubh", the default below, included) was rejected as "not compatible
# with model bulbul:v2," so v2 was temporarily made the default. Switched
# to a key with real v3 access; back to v3 by default. NOT independently
# re-verified against a live v3-enabled key from this environment (this
# repo's own dev key still lacks v3 access) -- confirm live once the real
# deployed key is updated. Kept model-aware (matching this file's own
# _PACE_RANGES pattern) so a key without v3 access degrades to a real
# error rather than a wrong/incompatible speaker being silently sent.
_DEFAULT_SPEAKERS = {"bulbul:v2": "anushka", "bulbul:v3": "shubh"}


class TTSStreamError(Exception):
    """Raised when the TTS WebSocket session fails."""


class SarvamTTSClient:
    def __init__(
        self,
        *,
        api_key_env: str = "SARVAM_API_KEY",
        ws_url: str | None = None,
        connect: Callable[..., object] | None = None,
    ):
        self._api_key_env = api_key_env
        self._ws_url = ws_url or os.environ.get("SARVAM_TTS_WS_URL", _DEFAULT_WS_URL)
        self._connect = connect or websockets.connect

    def _api_key(self) -> str:
        key = os.environ.get(self._api_key_env)
        if not key:
            raise TTSStreamError(f"{self._api_key_env} not set")
        return key

    @staticmethod
    def _config_data(
        language: str,
        model: str,
        *,
        voice: str | None,
        pace: float | None,
        pitch: float | None = None,
        loudness: float | None = None,
        temperature: float | None = None,
    ) -> dict:
        if model not in _PACE_RANGES:
            raise ValueError(f"unknown TTS model: {model!r}")
        lo, hi = _PACE_RANGES[model]
        if pace is not None and not (lo <= pace <= hi):
            raise ValueError(f"pace {pace} out of range for {model} ({lo}-{hi})")

        data: dict = {
            "target_language_code": _LANGUAGE_TO_SARVAM_CODE.get(language, language),
            "speaker": voice or _DEFAULT_SPEAKERS[model],
            "model": model,
            "output_audio_codec": _OUTPUT_AUDIO_CODEC,
            "speech_sample_rate": _SPEECH_SAMPLE_RATE,
        }
        if pace is not None:
            data["pace"] = pace
        if model == "bulbul:v2":
            if pitch is not None:
                data["pitch"] = pitch
            if loudness is not None:
                data["loudness"] = loudness
        else:  # bulbul:v3 — temperature, not pitch/loudness
            if temperature is not None:
                data["temperature"] = temperature
        return data

    async def synthesize(
        self,
        text_chunks: AsyncIterator[str],
        *,
        language: str,
        model: str = "bulbul:v3",
        voice: str | None = None,
        pace: float | None = None,
    ) -> AsyncIterator[bytes]:
        config_data = self._config_data(language, model, voice=voice, pace=pace)
        # Config itself carries no user text/PII -- safe to log in full, and
        # this is exactly the payload Sarvam's own support needs if a request
        # ID from this session ever needs escalating with them directly.
        logger.info("Sarvam TTS config: %s", config_data)
        try:
            async with self._connect(
                self._ws_url, additional_headers={"Authorization": f"Bearer {self._api_key()}"}
            ) as ws:
                await ws.send(json.dumps({"type": "config", "data": config_data}))

                async for text in text_chunks:
                    logger.info("Sarvam TTS sending text chunk (%d chars): %s", len(text), mask_pii(text))
                    await ws.send(json.dumps({"type": "text", "data": {"text": text}}))
                    # Sarvam buffers text server-side (min_buffer_size, default
                    # 50 chars) and only synthesizes on a flush — our own
                    # chunker deliberately sends short chunks (5-10 words) for
                    # fast TTFB, which is routinely under that threshold.
                    # Without an explicit flush per chunk, the server sits
                    # waiting for more buffered input while we sit waiting for
                    # its response — a real deadlock, found by testing this
                    # against the live API (it surfaces as a 408 "left open
                    # without any messages for too long").
                    await ws.send(json.dumps({"type": "flush"}))

                    received_any_audio = False
                    while True:
                        try:
                            raw = await asyncio.wait_for(
                                ws.recv(), timeout=_CHUNK_IDLE_TIMEOUT_SECONDS if received_any_audio else 15.0
                            )
                        except asyncio.TimeoutError:
                            logger.warning(
                                "Sarvam TTS idle timeout waiting for a response to this chunk "
                                "(received_any_audio=%s) -- treating as end of this chunk.",
                                received_any_audio,
                            )
                            break  # idle after audio — treat as this chunk's end (see module docstring)
                        event = json.loads(raw)
                        event_type = event.get("type")
                        # Real gap: a live "zero audio chunks, no exception"
                        # report couldn't be explained by any existing log --
                        # nothing recorded WHAT Sarvam actually sent back when
                        # it wasn't "audio". Logs every event TYPE (never the
                        # base64 audio payload itself -- that's the one case
                        # excluded, to avoid flooding logs with audio data).
                        if event_type == "audio":
                            logger.debug("Sarvam TTS event: type=audio (%d b64 chars)", len(event.get("data", {}).get("audio", "")))
                        else:
                            logger.info("Sarvam TTS event: type=%s data=%s", event_type, event.get("data"))
                        if event_type == "audio":
                            received_any_audio = True
                            yield base64.b64decode(event["data"]["audio"])
                        elif event_type == "error":
                            raise TTSStreamError(f"Sarvam TTS error: {event.get('data')}")
                        elif event_type == "event" and event.get("data", {}).get("event_type") == "final":
                            logger.info("Sarvam TTS: received explicit 'final' completion event")
                            break  # a real completion event, when Sarvam does send one
        except websockets.exceptions.ConnectionClosed as e:
            logger.warning(
                "Sarvam TTS WebSocket closed -- code=%s reason=%s",
                getattr(e, "code", None), getattr(e, "reason", None),
            )
            raise TTSStreamError(f"TTS stream connection closed: {e}") from e
        except OSError as e:
            logger.warning("Sarvam TTS WebSocket connection failed: %s", e)
            raise TTSStreamError(f"TTS stream connection failed: {e}") from e
