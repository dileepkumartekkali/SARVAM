"""Sarvam Saaras v3 streaming STT — the concrete SpeechSTTClient adapter.

Message protocol confirmed against Sarvam's live docs
(docs.sarvam.ai/api-reference-docs/speech-to-text/transcribe/ws, fetched July
2026) — this replaced an earlier best-guess protocol (a JSON `config`
message after connect, raw binary audio frames, `Authorization: Bearer`
auth) that was **proven wrong** by a real connection attempt: Sarvam
accepted the connection but rejected the first audio frame with
`Invalid request: 'audio' must not be None`, because the real API expects
config as URL query params (not a post-connect message), a
`Api-Subscription-Key` header (not `Authorization: Bearer`), and audio as
base64-encoded JSON messages (not raw binary WS frames). Sarvam's own event
shape (`{"type":"data",...}` / `{"type":"events",...}`) is translated here
into this module's stable internal shape
(`{"type":"transcript",...}` / `{"type":"vad","signal":...}`) so nothing
downstream (the gateway, tests) has to know the real wire format changed.

Reconnect-with-backoff for a dropped WS lives at the gateway, not here: once
this client has started forwarding audio from an `AsyncIterator[bytes]`, that
iterator can't be "rewound" if the connection dies mid-stream — the gateway
is the layer that holds a rolling raw-audio buffer and can fall back to
`transcribe_rest()` on that buffer (S2S plan §5). This client makes one
connection attempt per `stream()` call and raises `SpeechStreamError` clearly
on failure, rather than silently retrying past state it doesn't have.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
from typing import AsyncIterator, Callable
from urllib.parse import urlencode

import httpx
import websockets
import websockets.exceptions

from .clients import STTEvent, STTMode

_DEFAULT_WS_URL = "wss://api.sarvam.ai/speech-to-text/ws"
_DEFAULT_REST_URL = "https://api.sarvam.ai/speech-to-text"

# Once the audio source ends, how long to keep waiting for Sarvam's final
# transcript before giving up. Live behavior: the final transcript follows
# end-of-audio within ~1-2s when it comes at all.
_POST_AUDIO_FINAL_EVENTS_TIMEOUT_SECONDS = 5.0

# Sarvam's STT language codes are BCP-47 ("<lang>-IN"), not the bare 2-letter
# codes language_agent uses elsewhere in this codebase. "unknown" requests
# Sarvam's own auto-detect — used when no hint is available.
_LANGUAGE_TO_SARVAM_CODE = {
    "hi": "hi-IN", "bn": "bn-IN", "ta": "ta-IN", "te": "te-IN", "gu": "gu-IN",
    "kn": "kn-IN", "ml": "ml-IN", "mr": "mr-IN", "pa": "pa-IN", "or": "od-IN",
    "en": "en-IN",
}


class SpeechStreamError(Exception):
    """Raised when the STT WebSocket session fails. The gateway decides
    whether to retry, fall back to REST, or surface the failure."""


class SarvamSTTClient:
    def __init__(
        self,
        *,
        api_key_env: str = "SARVAM_API_KEY",
        ws_url: str | None = None,
        rest_url: str | None = None,
        connect: Callable[..., object] | None = None,
        http_transport: httpx.BaseTransport | None = None,
        timeout: float = 30.0,
        language_hint: str | None = None,
    ):
        self._api_key_env = api_key_env
        self._ws_url = ws_url or os.environ.get("SARVAM_STT_WS_URL", _DEFAULT_WS_URL)
        self._rest_url = rest_url or os.environ.get("SARVAM_STT_REST_URL", _DEFAULT_REST_URL)
        self._connect = connect or websockets.connect
        self._http_transport = http_transport
        self._timeout = timeout
        self._language_hint = language_hint

    def _api_key(self) -> str:
        key = os.environ.get(self._api_key_env)
        if not key:
            raise SpeechStreamError(f"{self._api_key_env} not set")
        return key

    async def stream(
        self,
        audio: AsyncIterator[bytes],
        *,
        codec: str,
        sample_rate: int = 16000,
        mode: STTMode = STTMode.CODEMIX,
        vad_signals: bool = True,
        high_vad_sensitivity: bool = True,
    ) -> AsyncIterator[STTEvent]:
        language_code = _LANGUAGE_TO_SARVAM_CODE.get(self._language_hint or "", "unknown")
        query = urlencode(
            {
                "language-code": language_code,
                "model": "saaras:v3",
                "mode": mode.value,
                "sample_rate": sample_rate,
                "high_vad_sensitivity": str(high_vad_sensitivity).lower(),
                "vad_signals": str(vad_signals).lower(),
                "input_audio_codec": codec,
            }
        )
        url = f"{self._ws_url}?{query}"

        try:
            async with self._connect(url, additional_headers={"Api-Subscription-Key": self._api_key()}) as ws:
                send_task = asyncio.ensure_future(self._pump_audio(ws, audio, sample_rate))
                try:
                    # A plain `async for raw in ws` here waited on Sarvam's
                    # socket INDEFINITELY after the audio source ended —
                    # observed live as `stt.stream` tracing spans that never
                    # completed, each pinning a Sarvam connection until
                    # Sarvam's own ~60s idle close. While audio is still
                    # flowing, recv() is raced against the pump finishing (a
                    # recv() entered before the pump ends must not block past
                    # it); once the pump is done, only a bounded window
                    # remains for Sarvam's final transcript.
                    while True:
                        try:
                            if send_task.done():
                                raw = await asyncio.wait_for(
                                    ws.recv(), timeout=_POST_AUDIO_FINAL_EVENTS_TIMEOUT_SECONDS
                                )
                            else:
                                recv_task = asyncio.ensure_future(ws.recv())
                                await asyncio.wait(
                                    {recv_task, send_task}, return_when=asyncio.FIRST_COMPLETED
                                )
                                if not recv_task.done():
                                    recv_task.cancel()
                                    continue  # pump just ended — re-enter in bounded mode
                                raw = recv_task.result()
                        except asyncio.TimeoutError:
                            break
                        except websockets.exceptions.ConnectionClosedOK:
                            break  # Sarvam closed cleanly — end of stream, not a failure
                        event = json.loads(raw)
                        translated = self._translate_event(event)
                        if translated is not None:
                            yield translated
                finally:
                    send_task.cancel()
        except websockets.exceptions.ConnectionClosed as e:
            raise SpeechStreamError(f"STT stream connection closed: {e}") from e
        except OSError as e:
            raise SpeechStreamError(f"STT stream connection failed: {e}") from e

    @staticmethod
    def _translate_event(event: dict) -> STTEvent | None:
        """Sarvam's real wire shape -> this module's stable internal shape."""
        event_type = event.get("type")
        data = event.get("data", {})
        if event_type == "data":
            return {
                "type": "transcript",
                "text": data.get("transcript", ""),
                "is_final": True,
                "confidence": data.get("language_probability"),
            }
        if event_type == "events":
            signal_type = data.get("signal_type")
            if signal_type == "START_SPEECH":
                return {"type": "vad", "signal": "speech_start"}
            if signal_type == "END_SPEECH":
                return {"type": "vad", "signal": "speech_end"}
            return None
        if event_type == "error":
            return {"type": "error", "reason": data.get("error", "unknown Sarvam STT error")}
        return None

    @staticmethod
    async def _pump_audio(ws, audio: AsyncIterator[bytes], sample_rate: int) -> None:
        async for chunk in audio:
            payload = {
                "audio": {
                    "data": base64.b64encode(chunk).decode("ascii"),
                    "sample_rate": str(sample_rate),
                    "encoding": "audio/wav",
                }
            }
            await ws.send(json.dumps(payload))

    async def transcribe_rest(self, audio: bytes, *, mode: STTMode = STTMode.CODEMIX) -> STTEvent:
        headers = {"Api-Subscription-Key": self._api_key()}
        async with httpx.AsyncClient(timeout=self._timeout, transport=self._http_transport) as client:
            resp = await client.post(
                self._rest_url,
                headers=headers,
                data={"mode": mode.value},
                files={"file": ("audio.wav", audio, "audio/wav")},
            )
            if resp.status_code != 200:
                raise SpeechStreamError(f"STT REST returned {resp.status_code}: {resp.text}")
            raw = resp.json()
            # Translated into the SAME stable shape `_translate_event` produces for
            # the streaming path — this was previously passed through as Sarvam's
            # raw REST response, which has no "type"/"is_final" keys at all (its
            # transcript field is named "transcript", matching the streaming
            # payload's "data.transcript"). A caller that only recognizes
            # `{"type": "transcript", "is_final": True}` (as the gateway and every
            # client do) never saw this fallback path as a real, final result.
            return {
                "type": "transcript",
                "text": raw.get("transcript", raw.get("text", "")),
                "is_final": True,
                "confidence": raw.get("language_probability"),
            }

    async def transcribe_batch(self, audio_uri: str, *, mode: STTMode = STTMode.CODEMIX) -> str:
        headers = {"Api-Subscription-Key": self._api_key()}
        async with httpx.AsyncClient(timeout=self._timeout, transport=self._http_transport) as client:
            resp = await client.post(
                f"{self._rest_url}/batch", headers=headers, json={"audio_uri": audio_uri, "mode": mode.value}
            )
            if resp.status_code != 200:
                raise SpeechStreamError(f"STT batch submission returned {resp.status_code}: {resp.text}")
            return resp.json()["job_id"]
