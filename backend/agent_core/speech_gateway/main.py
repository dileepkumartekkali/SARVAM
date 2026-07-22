"""Speech Gateway FastAPI app — runs as its own container/process
(Dockerfile.gateway), separate from the main backend. Holds Sarvam/Azure API
keys; the client never sees them and never talks to Sarvam directly.

Two independent WebSocket capabilities this phase (no full duplex — see
`duplex_session.py` for that, Phase 5):

  /ws/stt  client mic audio  -> gateway validates + proxies -> Sarvam STT
                              -> transcript/VAD events back to client.
                              Drops mid-utterance retry with backoff, then
                              fall back to REST on buffered audio (S2S §5).
  /ws/tts  client text deltas -> gateway chunks (agent_core.speech.chunker)
                              -> Sarvam or Azure-fallback TTS (one socket for
                                 the whole utterance) -> audio bytes to client.
                              Retries once on a fresh socket, then a
                              text-only fallback signal (S2S §5).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections import deque

from fastapi import Depends, FastAPI, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from ..observability.logging_config import configure_logging
from ..observability.metrics import (
    active_voice_sessions,
    active_websocket_connections,
    errors_total,
    metrics_response,
    reconnects_total,
    stt_latency_seconds,
    tts_ttfb_seconds,
)
from ..observability.tracing import init_tracing, start_span
from ..security.pii import mask_pii
from ..security.rate_limit import SlidingWindowRateLimiter
from ..speech.audio_validation import validate_pcm_frame
from ..speech.chunker import chunk_stream
from ..speech.clients import SpeechSTTClient, SpeechTTSClient, STTMode
from ..speech.fallback_tts import AzureFallbackTTSClient
from ..speech.sarvam_stt import SarvamSTTClient, SpeechStreamError
from ..speech.sarvam_tts import SarvamTTSClient, TTSStreamError
from ..speech.tts_provider_policy import select_tts_provider
from ..supervisor.session_state_machine import SessionPhase
from .backend_client import BackendChatError, BackendChatReply, call_backend_chat
from .duplex_session import DuplexSession
from .failure_policy import (
    LLM_TIMEOUT_APOLOGY,
    LOW_CONFIDENCE_TRANSCRIPT_QUESTION,
    TTS_FAILURE_APOLOGY,
    is_low_confidence_transcript,
    stt_stream_with_backoff_then_rest,
    tts_synthesize_with_retry_then_fallback,
)

configure_logging()
init_tracing("speech_gateway")
logger = logging.getLogger("agent_core.speech_gateway")

gateway_app = FastAPI(title="MAAV Speech Gateway", version="0.1.0")

_allowed_origins = [o for o in os.environ.get("CORS_ALLOWED_ORIGINS", "").split(",") if o]
gateway_app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["GET"],
    allow_headers=["Authorization"],
)

# Constructing these never touches the network (API keys are only read when a
# call is actually made) — same invariant as the main backend's router.
_sarvam_stt = SarvamSTTClient()
_sarvam_tts = SarvamTTSClient()
_fallback_tts = AzureFallbackTTSClient()

# S2S plan §5: on a dropped STT stream, fall back to REST on buffered audio
# rather than losing the utterance. Bounded to the last ~2s of 32ms frames.
_STT_AUDIO_BUFFER_FRAMES = 64
_STT_BACKOFF_MAX_SECONDS = 2.0

# A live client always sends its text + "__END__" within a couple seconds of
# opening the socket. Past this with no next message, the connection is
# almost certainly dead without having sent a clean close frame.
_CLIENT_IDLE_TIMEOUT_SECONDS = 10.0

# Voice sessions cost more than text turns — the cost-exhaustion abuse surface
# is bigger, so session *creation* (not every frame) is rate-limited per
# client IP. Env-configurable so a load test (or a deployment with known
# higher legitimate traffic) can tune it without a code change; single-
# process/in-memory — see security/rate_limit.py's note on multi-replica scaling.
_session_rate_limiter = SlidingWindowRateLimiter(
    max_requests=int(os.environ.get("STT_SESSION_RATE_LIMIT", "10")),
    window_seconds=float(os.environ.get("STT_SESSION_RATE_WINDOW_SECONDS", "60")),
)


_LOAD_TEST_MODE = os.environ.get("MAAV_LOAD_TEST_MODE", "").lower() == "true"
if _LOAD_TEST_MODE:
    # Never true in production — see load_test_fakes.py's docstring for why
    # this exists at all: there's no way to load-test 100-10,000 concurrent
    # sessions against a real (rate-limited, paid, third-party) Sarvam
    # account, so this measures the gateway's own connection-handling
    # ceiling instead.
    from .load_test_fakes import LoadTestFakeSTT, LoadTestFakeTTS

    _load_test_stt = LoadTestFakeSTT()
    _load_test_tts = LoadTestFakeTTS()


def get_stt_client() -> SpeechSTTClient:
    """FastAPI dependency, overridden in tests to inject a fake Sarvam client."""
    return _load_test_stt if _LOAD_TEST_MODE else _sarvam_stt


def get_tts_client(language: str) -> SpeechTTSClient:
    """Routes to Sarvam or the Azure fallback per tts_provider_policy — this
    is the "routed transparently by the Language Agent" language selection
    the S2S plan calls for; the client never sees which provider served it.
    """
    if _LOAD_TEST_MODE:
        return _load_test_tts
    return _sarvam_tts if select_tts_provider(language) == "sarvam" else _fallback_tts


async def _safe_close(websocket: WebSocket, *, code: int = 1000, reason: str = "") -> None:
    """Closing a WebSocket that the client (or uvicorn, on disconnect) already
    closed raises `RuntimeError` — a race that only shows up at high
    concurrency (found via the load test: docs/LOAD_TEST_REPORT.md). Every
    close in this module goes through here instead of `websocket.close()`
    directly.
    """
    try:
        await websocket.close(code=code, reason=reason)
    except RuntimeError:
        pass


@gateway_app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@gateway_app.get("/metrics")
async def metrics() -> Response:
    body, content_type = metrics_response()
    return Response(content=body, media_type=content_type)


def get_tts_client_resolver():
    """FastAPI dependency, overridden in tests to inject a fake TTS client
    regardless of the requested language."""
    return get_tts_client


def get_chat_caller():
    """FastAPI dependency, overridden in tests to inject a fake backend call
    without a real HTTP round trip."""
    return call_backend_chat


@gateway_app.websocket("/ws/stt")
async def stt_ws(websocket: WebSocket, stt_client: SpeechSTTClient = Depends(get_stt_client)):
    client_key = websocket.client.host if websocket.client else "unknown"
    if not _session_rate_limiter.allow(client_key):
        errors_total.labels(stage="rate_limit").inc()
        await _safe_close(websocket, code=1008, reason="rate limit exceeded")
        return

    await websocket.accept()
    active_voice_sessions.inc()
    active_websocket_connections.labels(kind="stt").inc()
    try:
        config = json.loads(await websocket.receive_text())
        codec = config["codec"]
        sample_rate = config.get("sample_rate", 16000)
        mode = STTMode(config.get("mode", STTMode.CODEMIX.value))
    except (WebSocketDisconnect, json.JSONDecodeError, KeyError, ValueError):
        active_websocket_connections.labels(kind="stt").dec()
        await _safe_close(websocket, code=1008, reason="invalid STT config")
        return

    audio_buffer: deque[bytes] = deque(maxlen=_STT_AUDIO_BUFFER_FRAMES)
    turn_start = time.monotonic()
    first_transcript_seen = False

    async def client_audio():
        while True:
            try:
                # Same dead-client leak class fixed in tts_ws: a live mic
                # sends a frame every 32ms, so this long with nothing at all
                # means the client died without a clean close frame — end
                # the stream instead of holding the Sarvam connection open
                # indefinitely.
                message = await asyncio.wait_for(websocket.receive(), timeout=_CLIENT_IDLE_TIMEOUT_SECONDS)
            except asyncio.TimeoutError:
                return
            if message["type"] == "websocket.disconnect":
                return
            frame = message.get("bytes")
            if frame is None:
                continue
            result = validate_pcm_frame(frame, sample_rate=sample_rate)
            if not result.ok:
                await websocket.send_json({"type": "error", "reason": result.reason})
                continue
            audio_buffer.append(frame)
            yield frame

    audio_gen = client_audio()  # one live generator, reused across retries — see failure_policy.py

    def stream_attempt():
        return stt_client.stream(audio_gen, codec=codec, sample_rate=sample_rate, mode=mode)

    async def rest_fallback():
        return await stt_client.transcribe_rest(b"".join(audio_buffer), mode=mode)

    def on_retry():
        reconnects_total.labels(stage="stt").inc()

    try:
        with start_span("speech_gateway", "stt.stream", sample_rate=sample_rate, mode=mode.value):
            async for event in stt_stream_with_backoff_then_rest(
                stream_attempt, rest_fallback, max_backoff_seconds=_STT_BACKOFF_MAX_SECONDS, on_retry=on_retry
            ):
                if event.get("type") == "transcript" and not first_transcript_seen:
                    first_transcript_seen = True
                    stt_latency_seconds.observe(time.monotonic() - turn_start)
                if event.get("type") == "transcript" and event.get("text"):
                    # PII-masked before it ever hits a log line — never log the raw transcript.
                    logger.debug("transcript event: %s", mask_pii(event["text"]))
                if event.get("type") == "error":
                    errors_total.labels(stage="stt").inc()
                try:
                    await websocket.send_json(event)
                except RuntimeError:
                    # The client got what it needed (its final transcript) and
                    # closed — a trailing event racing that close is normal
                    # turn-completion, not a failure worth a crashed handler
                    # and an ERROR span.
                    break
    except WebSocketDisconnect:
        pass
    finally:
        active_websocket_connections.labels(kind="stt").dec()
        await _safe_close(websocket)


@gateway_app.websocket("/ws/tts")
async def tts_ws(websocket: WebSocket, tts_client_resolver=Depends(get_tts_client_resolver)):
    # Real gap hit in a pre-deploy sweep: this was the only one of the three
    # voice-cost websockets (stt_ws, converse_ws both check) that accepted
    # unconditionally -- a client could open unlimited /ws/tts connections
    # and drive unbounded billed Sarvam synthesis calls, sidestepping the
    # exact abuse control rate_limit.py was built for.
    client_key = websocket.client.host if websocket.client else "unknown"
    if not _session_rate_limiter.allow(client_key):
        errors_total.labels(stage="rate_limit").inc()
        await _safe_close(websocket, code=1008, reason="rate limit exceeded")
        return

    await websocket.accept()
    active_websocket_connections.labels(kind="tts").inc()
    try:
        config = json.loads(await websocket.receive_text())
        language = config["language"]
        # An earlier key on this account had no bulbul:v3 access at all
        # (confirmed live, see agent_core/speech/sarvam_tts.py's own note) --
        # switched to a key with real v3 access; back to v3 by default. The
        # frontend also sends this explicitly (useVoiceSession.js).
        model = config.get("model", "bulbul:v3")
        voice = config.get("voice")
        pace = config.get("pace")
    except (WebSocketDisconnect, json.JSONDecodeError, KeyError):
        active_websocket_connections.labels(kind="tts").dec()
        await _safe_close(websocket, code=1008, reason="invalid TTS config")
        return

    tts_client = tts_client_resolver(language)
    turn_start = time.monotonic()
    first_chunk_seen = False

    async def client_text_deltas():
        while True:
            try:
                # A real bug found live: no timeout here meant a client whose
                # connection died without a clean close frame (a page reload,
                # a crash, a dropped network) left this `receive()` waiting
                # forever — the whole synthesis call, and the WS connection,
                # hung for as long as the process ran (observed: 20-45
                # minutes in tracing spans that never completed). A client
                # that's actually still there always sends its text +
                # "__END__" within a couple seconds, back to back.
                message = await asyncio.wait_for(websocket.receive(), timeout=_CLIENT_IDLE_TIMEOUT_SECONDS)
            except asyncio.TimeoutError:
                return
            if message["type"] == "websocket.disconnect":
                return
            raw = message.get("text")
            if raw is None:
                return
            # The client sends JSON envelopes: {"text": "..."} — found live
            # via an STT loopback whose transcript read "J text underscore
            # underscore...": this code was feeding the RAW envelope into
            # synthesis, so Sarvam was literally speaking the JSON wrapper
            # around every utterance. It also meant the {"text": "__END__"}
            # end marker never matched the bare-string comparison below, so
            # utterances only ever ended via disconnect or idle timeout.
            # A non-JSON payload still passes through as plain text.
            try:
                payload = json.loads(raw)
                text = payload.get("text") if isinstance(payload, dict) else raw
            except json.JSONDecodeError:
                text = raw
            if text is None or text == "__END__":
                return
            yield text

    _text_gen = chunk_stream(client_text_deltas())
    _sent_chunks: list[str] = []

    async def _replay_then_continue():
        # Real bug hit live, reproduced from a real report: primary_attempt()
        # used to hand the SAME _text_gen iterator to every retry attempt.
        # An async generator can only be consumed once -- if the first
        # attempt's socket died after sending one chunk (observed: Sarvam
        # closed the connection normally, code 1000, after ~5s of the TTS
        # socket sitting idle waiting for text while the LLM was still
        # generating it -- the socket is opened early, before any text
        # exists, for latency), the retry got whatever was LEFT of an
        # already-drained iterator -- for a short, one-sentence answer
        # that had already fully streamed by then, that was nothing at
        # all. Exactly matches the logged "zero audio chunks, no
        # exception" failure. Buffers every chunk as it's consumed so a
        # retry replays the FULL text from the start, then continues with
        # anything new, instead of silently sending less than before.
        for chunk in _sent_chunks:
            yield chunk
        async for chunk in _text_gen:
            _sent_chunks.append(chunk)
            yield chunk

    def primary_attempt():
        # One synthesize() call = one socket, reused for every chunk the
        # chunker yields — never opened per chunk (S2S plan §2-3).
        return tts_client.synthesize(_replay_then_continue(), language=language, model=model, voice=voice, pace=pace)

    async def on_text_only_fallback():
        errors_total.labels(stage="tts").inc()
        await websocket.send_json({"type": "text_only_fallback", "message": TTS_FAILURE_APOLOGY})

    # Real gap: nothing here ever logged how much audio actually got sent --
    # a request that "succeeds" (no exception) but Sarvam silently returns
    # zero chunks for (a valid config that just produces no audio for this
    # specific text/speaker/language combination) looked IDENTICAL in the
    # logs to one that worked, forcing guesswork on a live "no audio" report.
    chunk_count = 0
    total_bytes = 0
    try:
        with start_span("speech_gateway", "tts.synthesize", language=language, model=model):
            async for audio_bytes in tts_synthesize_with_retry_then_fallback(
                primary_attempt, on_text_only_fallback=on_text_only_fallback
            ):
                if not first_chunk_seen:
                    first_chunk_seen = True
                    tts_ttfb_seconds.observe(time.monotonic() - turn_start)
                chunk_count += 1
                total_bytes += len(audio_bytes)
                await websocket.send_bytes(audio_bytes)
        if chunk_count == 0:
            logger.warning(
                "TTS synthesis produced ZERO audio chunks with no exception raised "
                "(language=%s, model=%s, voice=%s) -- Sarvam accepted the request but "
                "silently returned no audio for this text/speaker/language combination.",
                language, model, voice,
            )
        else:
            logger.info(
                "TTS synthesis sent %d chunk(s), %d bytes (language=%s, model=%s, voice=%s)",
                chunk_count, total_bytes, language, model, voice,
            )
    except (SpeechStreamError, TTSStreamError) as e:
        errors_total.labels(stage="tts").inc()
        await websocket.send_json({"type": "error", "reason": str(e)})
    except WebSocketDisconnect:
        pass
    finally:
        active_websocket_connections.labels(kind="tts").dec()
        await _safe_close(websocket)


@gateway_app.websocket("/ws/converse")
async def converse_ws(
    websocket: WebSocket,
    stt_client: SpeechSTTClient = Depends(get_stt_client),
    tts_client_resolver=Depends(get_tts_client_resolver),
    chat_caller=Depends(get_chat_caller),
):
    """Full-duplex Speech<->Speech (S2S plan §3): the ONE endpoint that wires
    STT -> backend `/chat` -> TTS into a continuous multi-turn session.

    A single Sarvam STT stream stays open for the session's whole lifetime —
    not just per-utterance — so a `speech_start` VAD signal arriving while a
    reply is being thought up or spoken is recognized as a barge-in
    (`DuplexSession.handle_vad_speech_start`) rather than mistaken for the
    start of a brand new turn. Each finished utterance is relayed to the
    backend over HTTP exactly like a text client would; the reply is spoken
    back through TTS. See `backend_client.py` for why that's an HTTP call and
    not a direct import (separate processes, separate key sets).
    """
    client_key = websocket.client.host if websocket.client else "unknown"
    if not _session_rate_limiter.allow(client_key):
        errors_total.labels(stage="rate_limit").inc()
        await _safe_close(websocket, code=1008, reason="rate limit exceeded")
        return

    await websocket.accept()
    active_voice_sessions.inc()
    active_websocket_connections.labels(kind="converse").inc()

    try:
        config = json.loads(await websocket.receive_text())
        codec = config["codec"]
        sample_rate = config.get("sample_rate", 16000)
        stt_mode = STTMode(config.get("mode", STTMode.CODEMIX.value))
        language = config.get("language", "en-IN")
        session_id = config["session_id"]
        conversation_id = config["conversation_id"]
        thread_id = config["thread_id"]
        auth_token = config.get("auth_token", "")
    except (WebSocketDisconnect, json.JSONDecodeError, KeyError, ValueError):
        active_websocket_connections.labels(kind="converse").dec()
        await _safe_close(websocket, code=1008, reason="invalid converse config")
        return

    duplex = DuplexSession()
    # Set by a turn that got `pending_confirmation` back; forwarded on the
    # NEXT turn so the backend's ConfirmationGate can consume it if (and only
    # if) the model reissues the exact same tool+args — see
    # backend_client.call_backend_chat's docstring.
    pending_confirmation_token: str | None = None
    turn_task: asyncio.Task | None = None

    async def client_audio():
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                return
            frame = message.get("bytes")
            if frame is None:
                continue
            if validate_pcm_frame(frame, sample_rate=sample_rate).ok:
                yield frame

    # STT runs as its own long-lived task feeding a queue, rather than being
    # iterated inline, so the orchestrator loop below stays free to notice a
    # barge-in VAD signal even while a turn (think + speak) is in flight.
    event_queue: asyncio.Queue = asyncio.Queue()

    async def stt_pump() -> None:
        try:
            async for event in stt_client.stream(
                client_audio(), codec=codec, sample_rate=sample_rate, mode=stt_mode
            ):
                await event_queue.put(event)
        except WebSocketDisconnect:
            pass
        except Exception as e:  # noqa: BLE001 — surfaced to the client as a stream error, not a crash
            errors_total.labels(stage="stt").inc()
            await event_queue.put({"type": "error", "reason": str(e)})
        await event_queue.put({"type": "_stt_stream_ended"})

    stt_task = asyncio.ensure_future(stt_pump())

    async def think(transcript: str) -> BackendChatReply:
        with start_span("speech_gateway", "converse.think"):
            return await chat_caller(
                message=transcript,
                session_id=session_id,
                conversation_id=conversation_id,
                thread_id=thread_id,
                language=language,
                auth_token=auth_token,
                confirmation_token=pending_confirmation_token,
            )

    async def speak(text: str) -> None:
        tts_client = tts_client_resolver(language)

        async def one_chunk():
            yield text

        async def on_text_only_fallback():
            errors_total.labels(stage="tts").inc()
            await websocket.send_json({"type": "text_only_fallback", "message": TTS_FAILURE_APOLOGY})

        with start_span("speech_gateway", "converse.speak", language=language):
            async for audio_bytes in tts_synthesize_with_retry_then_fallback(
                lambda: tts_client.synthesize(one_chunk(), language=language),
                on_text_only_fallback=on_text_only_fallback,
            ):
                await websocket.send_bytes(audio_bytes)

    async def run_turn(transcript: str) -> None:
        nonlocal pending_confirmation_token

        think_task = asyncio.ensure_future(think(transcript))
        duplex.start_thinking(think_task)
        try:
            reply = await think_task
        except asyncio.CancelledError:
            return  # barge-in already reset state and sent barge_in_detected
        except BackendChatError as e:
            errors_total.labels(stage="backend_chat").inc()
            await websocket.send_json({"type": "error", "reason": str(e)})
            reply = BackendChatReply(text=LLM_TIMEOUT_APOLOGY)

        pending_confirmation_token = reply.pending_confirmation_token
        await websocket.send_json({"type": "assistant_text", "text": reply.text})

        speak_task = asyncio.ensure_future(speak(reply.text))
        duplex.start_speaking(speak_task)
        try:
            await speak_task
        except asyncio.CancelledError:
            return

        duplex.finish_turn()
        await websocket.send_json({"type": "turn_complete"})

    try:
        while True:
            event = await event_queue.get()
            event_type = event.get("type")

            if event_type == "_stt_stream_ended":
                break

            if event_type == "vad" and event.get("signal") == "speech_start":
                if await duplex.handle_vad_speech_start():
                    await websocket.send_json({"type": "barge_in_detected"})
                continue

            if event_type == "error":
                await websocket.send_json(event)
                continue

            if event_type != "transcript" or not event.get("is_final"):
                continue

            if duplex.state.phase != SessionPhase.LISTENING:
                continue  # a stray final transcript mid-turn (shouldn't happen) — ignore rather than overlap turns

            if is_low_confidence_transcript(event):
                await websocket.send_json({"type": "clarify", "text": LOW_CONFIDENCE_TRANSCRIPT_QUESTION})
                continue

            await websocket.send_json({"type": "transcript_final", "text": event.get("text", "")})
            turn_task = asyncio.ensure_future(run_turn(event["text"]))
    except WebSocketDisconnect:
        pass
    finally:
        stt_task.cancel()
        if turn_task is not None and not turn_task.done():
            turn_task.cancel()
        active_websocket_connections.labels(kind="converse").dec()
        await _safe_close(websocket)
