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

# Real bug hit live, confirmed by direct testing: the OLD synthesize() sent
# one chunk, then BLOCKED waiting up to this many seconds of dead silence on
# the socket before sending the NEXT chunk -- reproduced with the actual
# production code: a 3-sentence reply had 2.4s and ~4s silent gaps between
# sentences, matching this constant exactly. That's the direct cause of the
# "speech stops, pauses, then continues" symptom reported live.
#
# Root cause: `send_completion_event` DOES make Sarvam send a real `final`
# event -- the previous comment here claiming otherwise tested it the same
# broken way (waiting for one chunk's own completion before ever sending
# the next), which never gives Sarvam two chunks to batch together in the
# first place. Confirmed directly: `final` fires per FLUSHED BATCH, not
# once for the whole session -- if the very next chunk hasn't been sent yet
# when an earlier batch's `final` arrives, treating that as "fully done"
# would cut off every later sentence. synthesize() below now feeds all
# incoming chunks continuously (never waiting on completion of an earlier
# one before sending the next) and only treats `final`/idle-silence as
# real completion once the caller's OWN chunk iterator is exhausted -- so a
# `final` arriving early is correctly a no-op, not a stop signal.
_CHUNK_IDLE_TIMEOUT_SECONDS = 2.0
# How long to wait for the very first response after a flush, before any
# audio has arrived yet for this batch -- longer than the idle timeout
# above since a cold synthesis start can genuinely take a few seconds.
_INITIAL_RESPONSE_TIMEOUT_SECONDS = 15.0

_DEFAULT_WS_URL = "wss://api.sarvam.ai/text-to-speech/ws?send_completion_event=true"

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

# Real bug hit live, TWICE: every key tried on this account so far has
# been rejected by Sarvam's own server for bulbul:v3 -- "Speaker 'shubh'
# is not compatible with model bulbul:v2" -- confirmed directly in
# production logs, not a guess, even after a claimed key/plan change.
# "bulbul:v2" + "anushka" is the one combination proven, repeatedly, with
# real audio bytes returned. Kept model-aware (not a single flat default)
# so bulbul:v3 stays a correct, ready-to-use path the moment a genuinely
# v3-enabled key is confirmed -- see synthesize()'s own default and
# every call site (all currently "bulbul:v2"). Before ever switching the
# default back to v3, verify end-to-end with a real call like:
#   SarvamTTSClient().synthesize(<texts>, language="en", model="bulbul:v3")
# and confirm it returns real audio bytes, not a 422 -- do not just trust
# that a key/plan change was applied correctly.
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
            # Confirmed live: Sarvam accepts this and returns real audio.
            # Text normalization (numbers, abbreviations, punctuation) before
            # synthesis -- a real Twilio-based voice agent comparison project
            # sets this explicitly; MAAV never had before this.
            "enable_preprocessing": True,
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
        # Config itself carries no user text/PII -- safe to log in full, and
        # this is exactly the payload Sarvam's own support needs if a request
        # ID from this session ever needs escalating with them directly.
        logger.info("Sarvam TTS config: %s", config_data)
        try:
            # Switched from "Authorization: Bearer" -- confirmed live that
            # both schemes work on this account, but this one matches
            # sarvam_stt.py's own auth (Api-Subscription-Key), so TTS and
            # STT are no longer internally inconsistent with each other for
            # no reason.
            async with self._connect(
                self._ws_url, additional_headers={"Api-Subscription-Key": self._api_key()}
            ) as ws:
                await ws.send(json.dumps({"type": "config", "data": config_data}))

                # Feeds every incoming chunk to the socket continuously, never
                # waiting for an earlier chunk's own synthesis to finish first
                # -- that serialized wait was the actual cause of the
                # multi-second silent gaps between sentences (see
                # _CHUNK_IDLE_TIMEOUT_SECONDS's own comment). `send_done` is
                # the ONLY correct signal that no more text is coming (the
                # caller's own chunk iterator -- typically paced by the LLM's
                # own token generation -- is exhausted).
                send_done = asyncio.Event()
                send_error: BaseException | None = None
                # Real bug hit live in this fix's own first attempt: when
                # chunks are sent close together (the common case -- the LLM
                # already produced several sentences before this generator
                # even started consuming them), `_pump_text` can finish
                # sending (and set send_done) microseconds after the FIRST
                # chunk's own `final` event arrives, before the SECOND and
                # THIRD chunks' own audio/final have arrived at all --
                # `send_done.is_set()` alone can't tell "no more chunks are
                # coming" apart from "no more chunks are coming YET, but
                # earlier ones haven't finished." Counting flushes sent vs.
                # finals received distinguishes them correctly.
                chunks_flushed = 0
                finals_received = 0

                async def _pump_text() -> None:
                    nonlocal send_error, chunks_flushed
                    try:
                        async for text in text_chunks:
                            logger.info("Sarvam TTS sending text chunk (%d chars): %s", len(text), mask_pii(text))
                            await ws.send(json.dumps({"type": "text", "data": {"text": text}}))
                            # Sarvam buffers text server-side (min_buffer_size,
                            # default 50 chars) and only synthesizes on a flush
                            # -- our own chunker deliberately sends short chunks
                            # (5-10 words) for fast TTFB, routinely under that
                            # threshold. Without an explicit flush per chunk,
                            # the server sits waiting for more buffered input
                            # -- confirmed live (surfaces as a 408 "left open
                            # without any messages for too long").
                            # Real bug hit live in this fix's own first
                            # attempt: incrementing chunks_flushed AFTER
                            # awaiting send() left a real race -- the receive
                            # loop's own concurrent recv() can process a
                            # response to THIS flush before this coroutine
                            # resumes past its own await to increment the
                            # counter, undercounting chunks_flushed at the
                            # exact moment it's checked. Incrementing first
                            # means the counter always reflects "committed to
                            # sending," matching a real socket where the
                            # message is physically gone the instant send()
                            # is called, not after some later line runs.
                            chunks_flushed += 1
                            await ws.send(json.dumps({"type": "flush"}))
                    except BaseException as e:  # noqa: BLE001 -- surfaced via send_error, not swallowed
                        send_error = e
                    finally:
                        send_done.set()

                send_task = asyncio.ensure_future(_pump_text())
                try:
                    received_any_audio = False
                    while True:
                        try:
                            raw = await asyncio.wait_for(
                                ws.recv(),
                                timeout=_CHUNK_IDLE_TIMEOUT_SECONDS if received_any_audio else _INITIAL_RESPONSE_TIMEOUT_SECONDS,
                            )
                        except asyncio.TimeoutError:
                            if send_done.is_set() and finals_received >= chunks_flushed:
                                logger.warning(
                                    "Sarvam TTS idle timeout after the last chunk was sent "
                                    "(received_any_audio=%s) -- treating as end of the reply.",
                                    received_any_audio,
                                )
                                break
                            # Either more text is still coming (the LLM hasn't
                            # produced it yet), or an earlier flushed batch's
                            # own audio/final hasn't arrived yet -- a lack of
                            # a NEW event right now is normal, not a stall.
                            # Keep waiting instead of giving up on the rest of
                            # the reply.
                            received_any_audio = False
                            continue
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
                            # Real gap, twice over: (1) `final` fires per
                            # FLUSHED BATCH, not once for the whole session --
                            # if a later chunk hasn't produced its own final
                            # yet, this is NOT the end of the reply. (2) When
                            # chunks are sent close together (the common
                            # case), `_pump_text` can finish sending -- and
                            # set send_done -- microseconds after only the
                            # FIRST chunk's final arrives, well before later
                            # chunks' own finals do. send_done alone can't
                            # tell "nothing more is coming" apart from
                            # "nothing more is coming YET" -- only the finals
                            # actually received vs. batches actually flushed
                            # can.
                            finals_received += 1
                            if send_done.is_set() and finals_received >= chunks_flushed:
                                logger.info("Sarvam TTS: received explicit 'final' completion event")
                                break
                    if send_error is not None:
                        raise send_error
                finally:
                    if not send_task.done():
                        send_task.cancel()
                    try:
                        await send_task
                    except (asyncio.CancelledError, Exception):  # noqa: BLE001 -- already surfaced via send_error above if real
                        pass
        except websockets.exceptions.ConnectionClosed as e:
            logger.warning(
                "Sarvam TTS WebSocket closed -- code=%s reason=%s",
                getattr(e, "code", None), getattr(e, "reason", None),
            )
            raise TTSStreamError(f"TTS stream connection closed: {e}") from e
        except OSError as e:
            logger.warning("Sarvam TTS WebSocket connection failed: %s", e)
            raise TTSStreamError(f"TTS stream connection failed: {e}") from e
