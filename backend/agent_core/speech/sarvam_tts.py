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
import os
from typing import AsyncIterator, Callable

import websockets
import websockets.exceptions

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

# Real bug hit live, reported as "English is clear, Telugu is not,
# and now no audio at all": this app assumed model="bulbul:v3" as the
# default everywhere, with "shubh" as its speaker (bulbul:v3's own
# documented default). Confirmed DIRECTLY against the live API with this
# account's real key: every v3-only speaker (shubh, priya, aditya, ...) is
# rejected as "not compatible with model bulbul:v2" regardless of what
# "model" value is sent -- this account's real Sarvam API does not honor
# bulbul:v3 at all, it always evaluates as v2. "anushka" (a real v2
# speaker) succeeded in that same direct test. The actual default model is
# now "bulbul:v2" (see synthesize()'s own default, and every call site) --
# kept model-aware here (matching this file's own _PACE_RANGES pattern)
# so bulbul:v3 stays a supported, correct path if this account's access
# ever changes, without being the (currently broken) default.
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
        model: str = "bulbul:v2",
        voice: str | None = None,
        pace: float | None = None,
    ) -> AsyncIterator[bytes]:
        config_data = self._config_data(language, model, voice=voice, pace=pace)
        try:
            async with self._connect(
                self._ws_url, additional_headers={"Authorization": f"Bearer {self._api_key()}"}
            ) as ws:
                await ws.send(json.dumps({"type": "config", "data": config_data}))

                async for text in text_chunks:
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
                            break  # idle after audio — treat as this chunk's end (see module docstring)
                        event = json.loads(raw)
                        event_type = event.get("type")
                        if event_type == "audio":
                            received_any_audio = True
                            yield base64.b64decode(event["data"]["audio"])
                        elif event_type == "error":
                            raise TTSStreamError(f"Sarvam TTS error: {event.get('data')}")
                        elif event_type == "event" and event.get("data", {}).get("event_type") == "final":
                            break  # a real completion event, when Sarvam does send one
        except websockets.exceptions.ConnectionClosed as e:
            raise TTSStreamError(f"TTS stream connection closed: {e}") from e
        except OSError as e:
            raise TTSStreamError(f"TTS stream connection failed: {e}") from e
