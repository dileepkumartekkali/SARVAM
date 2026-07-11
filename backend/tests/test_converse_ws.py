"""Full-duplex `/ws/converse`: STT -> backend `/chat` -> TTS wired into one
session, driven through FastAPI's TestClient (in-process ASGI) with fake
STT/TTS/backend-call dependencies injected — no real network, no real Sarvam
or LLM calls."""

import asyncio
import json

from fastapi.testclient import TestClient

from agent_core.speech.clients import STTMode
from agent_core.speech_gateway.backend_client import BackendChatError, BackendChatReply
from agent_core.speech_gateway.main import (
    gateway_app,
    get_chat_caller,
    get_stt_client,
    get_tts_client_resolver,
)


def _valid_frame() -> bytes:
    expected_bytes = int(16000 * 0.032) * 2
    return b"\x00\x01" * (expected_bytes // 2)


class ScriptedGatewaySTT:
    """Yields one canned event per audio frame received, then blocks forever
    (a real mic never runs dry) so the test controls the pace entirely via
    how many frames it sends."""

    def __init__(self, events_per_frame):
        self._events_per_frame = events_per_frame

    async def stream(self, audio, *, codec, sample_rate=16000, mode=STTMode.CODEMIX, vad_signals=True, high_vad_sensitivity=True):
        i = 0
        async for _frame in audio:
            for event in self._events_per_frame[min(i, len(self._events_per_frame) - 1)]:
                yield event
            i += 1

    async def transcribe_rest(self, audio, *, mode=STTMode.CODEMIX):
        return {"text": "rest-fallback"}


class RecordingGatewayTTS:
    def __init__(self):
        self.synthesized: list[str] = []

    async def synthesize(self, text_chunks, *, language, model="bulbul:v3", voice=None, pace=None):
        async for chunk in text_chunks:
            self.synthesized.append(chunk)
            yield f"AUDIO[{chunk}]".encode()


def _converse_config(**overrides) -> str:
    config = {
        "codec": "pcm_s16le",
        "sample_rate": 16000,
        "language": "hi",
        "session_id": "s1",
        "conversation_id": "c1",
        "thread_id": "t1",
        "auth_token": "test-token",
    }
    config.update(overrides)
    return json.dumps(config)


def _override(overrides: dict):
    gateway_app.dependency_overrides.update(overrides)
    return gateway_app.dependency_overrides


def test_happy_path_turn_relays_transcript_and_speaks_reply():
    stt = ScriptedGatewaySTT(events_per_frame=[[{"type": "transcript", "text": "hello", "is_final": True, "confidence": 0.9}]])
    tts = RecordingGatewayTTS()
    calls = []

    async def fake_chat_caller(**kwargs):
        calls.append(kwargs)
        return BackendChatReply(text="Hi there!")

    _override(
        {
            get_stt_client: lambda: stt,
            get_tts_client_resolver: lambda: (lambda language: tts),
            get_chat_caller: lambda: fake_chat_caller,
        }
    )
    try:
        client = TestClient(gateway_app)
        with client.websocket_connect("/ws/converse") as ws:
            ws.send_text(_converse_config())
            ws.send_bytes(_valid_frame())

            transcript_msg = ws.receive_json()
            assistant_msg = ws.receive_json()
            audio = ws.receive_bytes()
            turn_complete = ws.receive_json()

        assert transcript_msg == {"type": "transcript_final", "text": "hello"}
        assert assistant_msg == {"type": "assistant_text", "text": "Hi there!"}
        assert audio == b"AUDIO[Hi there!]"
        assert turn_complete == {"type": "turn_complete"}
        assert calls[0]["message"] == "hello"
        assert calls[0]["auth_token"] == "test-token"
        assert calls[0]["confirmation_token"] is None
    finally:
        gateway_app.dependency_overrides.clear()


def test_low_confidence_transcript_asks_to_clarify_without_calling_backend():
    stt = ScriptedGatewaySTT(events_per_frame=[[{"type": "transcript", "text": "", "is_final": True, "confidence": 0.1}]])
    called = False

    async def fake_chat_caller(**kwargs):
        nonlocal called
        called = True
        return BackendChatReply(text="should never be reached")

    _override(
        {
            get_stt_client: lambda: stt,
            get_tts_client_resolver: lambda: (lambda language: RecordingGatewayTTS()),
            get_chat_caller: lambda: fake_chat_caller,
        }
    )
    try:
        client = TestClient(gateway_app)
        with client.websocket_connect("/ws/converse") as ws:
            ws.send_text(_converse_config())
            ws.send_bytes(_valid_frame())
            clarify_msg = ws.receive_json()

        assert clarify_msg["type"] == "clarify"
        assert called is False
    finally:
        gateway_app.dependency_overrides.clear()


def test_pending_confirmation_token_is_forwarded_on_the_next_turn():
    stt = ScriptedGatewaySTT(
        events_per_frame=[
            [{"type": "transcript", "text": "delete my note", "is_final": True, "confidence": 0.9}],
            [{"type": "transcript", "text": "yes confirm", "is_final": True, "confidence": 0.9}],
        ]
    )
    tts = RecordingGatewayTTS()
    calls = []

    async def fake_chat_caller(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return BackendChatReply(
                text="Say confirm to delete it.",
                pending_confirmation_token="tok-123",
                pending_confirmation_tool="delete_note",
            )
        return BackendChatReply(text="Deleted.")

    _override(
        {
            get_stt_client: lambda: stt,
            get_tts_client_resolver: lambda: (lambda language: tts),
            get_chat_caller: lambda: fake_chat_caller,
        }
    )
    try:
        client = TestClient(gateway_app)
        with client.websocket_connect("/ws/converse") as ws:
            ws.send_text(_converse_config())
            ws.send_bytes(_valid_frame())  # turn 1: triggers pending confirmation
            ws.receive_json()  # transcript_final
            ws.receive_json()  # assistant_text
            ws.receive_bytes()  # audio
            ws.receive_json()  # turn_complete

            ws.send_bytes(_valid_frame())  # turn 2: should carry the token forward
            ws.receive_json()
            ws.receive_json()
            ws.receive_bytes()
            ws.receive_json()

        assert calls[0]["confirmation_token"] is None
        assert calls[1]["confirmation_token"] == "tok-123"
    finally:
        gateway_app.dependency_overrides.clear()


def test_backend_chat_failure_speaks_a_fixed_apology_not_a_crash():
    stt = ScriptedGatewaySTT(events_per_frame=[[{"type": "transcript", "text": "hello", "is_final": True, "confidence": 0.9}]])
    tts = RecordingGatewayTTS()

    async def failing_chat_caller(**kwargs):
        raise BackendChatError("simulated backend timeout")

    _override(
        {
            get_stt_client: lambda: stt,
            get_tts_client_resolver: lambda: (lambda language: tts),
            get_chat_caller: lambda: failing_chat_caller,
        }
    )
    try:
        client = TestClient(gateway_app)
        with client.websocket_connect("/ws/converse") as ws:
            ws.send_text(_converse_config())
            ws.send_bytes(_valid_frame())

            ws.receive_json()  # transcript_final
            error_msg = ws.receive_json()
            assistant_msg = ws.receive_json()
            ws.receive_bytes()
            turn_complete = ws.receive_json()

        assert error_msg["type"] == "error"
        assert "getting an answer" in assistant_msg["text"]
        assert turn_complete == {"type": "turn_complete"}
    finally:
        gateway_app.dependency_overrides.clear()


def test_barge_in_during_speaking_cancels_tts_and_emits_barge_in_event():
    """A `speech_start` VAD signal arriving while the reply is being spoken
    must cancel TTS mid-stream and reset to LISTENING — proven end-to-end
    through the real websocket route, not just DuplexSession in isolation."""

    class SlowTTS:
        async def synthesize(self, text_chunks, *, language, model="bulbul:v3", voice=None, pace=None):
            async for _ in text_chunks:
                await asyncio.sleep(30)
                yield b"should never be reached"

    stt = ScriptedGatewaySTT(
        events_per_frame=[
            [{"type": "transcript", "text": "hello", "is_final": True, "confidence": 0.9}],
            [{"type": "vad", "signal": "speech_start"}],
        ]
    )

    async def fake_chat_caller(**kwargs):
        return BackendChatReply(text="a very long reply that takes a while to speak")

    _override(
        {
            get_stt_client: lambda: stt,
            get_tts_client_resolver: lambda: (lambda language: SlowTTS()),
            get_chat_caller: lambda: fake_chat_caller,
        }
    )
    try:
        client = TestClient(gateway_app)
        with client.websocket_connect("/ws/converse") as ws:
            ws.send_text(_converse_config())
            ws.send_bytes(_valid_frame())  # triggers the turn
            ws.receive_json()  # transcript_final
            ws.receive_json()  # assistant_text

            ws.send_bytes(_valid_frame())  # arrives while SlowTTS is "speaking" -> barge-in
            barge_in_msg = ws.receive_json()

        assert barge_in_msg == {"type": "barge_in_detected"}
    finally:
        gateway_app.dependency_overrides.clear()
