"""Speech Gateway WS routes, driven through FastAPI's TestClient
(in-process ASGI, no real socket/network) with fake STT/TTS clients injected
via dependency overrides."""

import json

from fastapi.testclient import TestClient

from agent_core.speech.clients import STTMode
from agent_core.speech_gateway.main import gateway_app, get_stt_client, get_tts_client_resolver


class FakeGatewaySTT:
    """Yields one canned event per audio frame received — simulates
    incremental VAD/transcript events as real-time as the fake can manage."""

    def __init__(self, events_per_frame):
        self._events_per_frame = events_per_frame
        self.received_frames: list[bytes] = []

    async def stream(self, audio, *, codec, sample_rate=16000, mode=STTMode.CODEMIX, vad_signals=True, high_vad_sensitivity=True):
        i = 0
        async for frame in audio:
            self.received_frames.append(frame)
            for event in self._events_per_frame[min(i, len(self._events_per_frame) - 1)]:
                yield event
            i += 1

    async def transcribe_rest(self, audio, *, mode=STTMode.CODEMIX):
        return {"text": "rest-fallback-transcript"}

    async def transcribe_batch(self, audio_uri, *, mode=STTMode.CODEMIX):
        return "job-1"


class FailingGatewaySTT:
    """Every connection attempt fails immediately, without touching the audio
    generator at all — models a connection-establishment failure (e.g. Sarvam
    unreachable) rather than a mid-stream drop, so retries don't block
    waiting for more frames a test client isn't continuously producing (a
    real mic would be)."""

    def __init__(self):
        self.rest_calls: list[bytes] = []
        self.attempts = 0

    async def stream(self, audio, *, codec, sample_rate=16000, mode=STTMode.CODEMIX, vad_signals=True, high_vad_sensitivity=True):
        from agent_core.speech.sarvam_stt import SpeechStreamError

        self.attempts += 1
        raise SpeechStreamError("simulated drop")
        yield  # pragma: no cover — unreachable, satisfies async generator shape

    async def transcribe_rest(self, audio, *, mode=STTMode.CODEMIX):
        self.rest_calls.append(audio)
        return {"text": "recovered via rest"}


class FakeGatewayTTS:
    def __init__(self):
        self.synthesize_calls: list[str] = []

    async def synthesize(self, text_chunks, *, language, model="bulbul:v3", voice=None, pace=None):
        async for chunk in text_chunks:
            self.synthesize_calls.append(chunk)
            yield f"AUDIO[{chunk}]".encode()


def _valid_frame() -> bytes:
    expected_bytes = int(16000 * 0.032) * 2
    return b"\x00\x01" * (expected_bytes // 2)


def test_stt_ws_streams_transcript_events_for_valid_frames():
    fake = FakeGatewaySTT(events_per_frame=[[{"type": "transcript", "text": "hello", "is_final": True}]])
    gateway_app.dependency_overrides[get_stt_client] = lambda: fake
    try:
        client = TestClient(gateway_app)
        with client.websocket_connect("/ws/stt") as ws:
            ws.send_text(json.dumps({"codec": "pcm_s16le", "sample_rate": 16000}))
            ws.send_bytes(_valid_frame())
            event = ws.receive_json()
        assert event == {"type": "transcript", "text": "hello", "is_final": True}
        assert fake.received_frames == [_valid_frame()]
    finally:
        gateway_app.dependency_overrides.clear()


def test_stt_ws_rejects_malformed_frame_without_forwarding_it():
    fake = FakeGatewaySTT(events_per_frame=[[]])
    gateway_app.dependency_overrides[get_stt_client] = lambda: fake
    try:
        client = TestClient(gateway_app)
        with client.websocket_connect("/ws/stt") as ws:
            ws.send_text(json.dumps({"codec": "pcm_s16le", "sample_rate": 16000}))
            ws.send_bytes(b"\x01")  # odd byte count — invalid PCM16
            error = ws.receive_json()
        assert error["type"] == "error"
        assert fake.received_frames == []  # never forwarded to the STT client
    finally:
        gateway_app.dependency_overrides.clear()


def test_stt_ws_falls_back_to_rest_on_stream_failure():
    fake = FailingGatewaySTT()
    gateway_app.dependency_overrides[get_stt_client] = lambda: fake
    try:
        client = TestClient(gateway_app)
        with client.websocket_connect("/ws/stt") as ws:
            ws.send_text(json.dumps({"codec": "pcm_s16le", "sample_rate": 16000}))
            ws.send_bytes(_valid_frame())
            result = ws.receive_json()
        assert result["via"] == "rest_fallback"
        assert result["text"] == "recovered via rest"
        assert len(fake.rest_calls) == 1
    finally:
        gateway_app.dependency_overrides.clear()


def test_tts_ws_synthesizes_the_inner_text_not_the_json_envelope():
    """A real bug found live via STT loopback (a transcript read "J text
    underscore underscore..."): the route fed the client's RAW {"text": ...}
    JSON envelope into synthesis, so the TTS voice was literally speaking
    the JSON wrapper around every utterance. The old version of this test
    used a substring assertion (`b"Hello there." in chunk`), which passed
    either way and masked it — these assertions pin the exact text
    synthesized."""
    fake = FakeGatewayTTS()
    gateway_app.dependency_overrides[get_tts_client_resolver] = lambda: (lambda language: fake)
    try:
        client = TestClient(gateway_app)
        with client.websocket_connect("/ws/tts") as ws:
            ws.send_text(json.dumps({"language": "hi", "model": "bulbul:v3"}))
            ws.send_text(json.dumps({"text": "Hello there. "}))
            ws.send_text(json.dumps({"text": "Second sentence."}))
            ws.send_text(json.dumps({"text": "__END__"}))
            # chunk_stream yields exactly two sentence chunks for this input.
            audio_chunks = [ws.receive_bytes(), ws.receive_bytes()]
        assert fake.synthesize_calls == ["Hello there.", "Second sentence."]
        assert audio_chunks[0] == b"AUDIO[Hello there.]"
    finally:
        gateway_app.dependency_overrides.clear()


def test_tts_ws_closes_after_client_goes_idle_without_end_signal(monkeypatch):
    """A real bug found live: waiting for the client's next text delta had no
    timeout — a connection that died without a clean close frame (a page
    reload, a crash, a dropped network) left the whole synthesis call
    hanging indefinitely. Observed in production tracing as 20-45 minute
    unclosed spans. Must not hang here either."""
    import agent_core.speech_gateway.main as gateway_main

    monkeypatch.setattr(gateway_main, "_CLIENT_IDLE_TIMEOUT_SECONDS", 0.05)
    fake = FakeGatewayTTS()
    gateway_app.dependency_overrides[get_tts_client_resolver] = lambda: (lambda language: fake)
    try:
        client = TestClient(gateway_app)
        with client.websocket_connect("/ws/tts") as ws:
            ws.send_text(json.dumps({"language": "hi", "model": "bulbul:v3"}))
            # Never sends text or "__END__" — the connection must close on
            # its own after the idle timeout, not hang waiting forever.
            closed = False
            try:
                ws.receive_bytes()
            except Exception:
                closed = True
            assert closed
    finally:
        gateway_app.dependency_overrides.clear()


def test_tts_ws_routes_assamese_to_fallback_via_policy():
    """Not overriding the resolver — proves the real select_tts_provider
    policy is what the route consults for an unsupported-by-Sarvam language."""
    from agent_core.speech_gateway.main import get_tts_client

    resolved = get_tts_client("as")
    from agent_core.speech.fallback_tts import AzureFallbackTTSClient

    assert isinstance(resolved, AzureFallbackTTSClient)


def test_rate_limit_rejection_increments_error_metric():
    from agent_core.observability.metrics import errors_total
    from agent_core.speech_gateway.main import _session_rate_limiter

    def rate_limit_error_count():
        for metric in errors_total.collect():
            for sample in metric.samples:
                if sample.labels.get("stage") == "rate_limit" and sample.name.endswith("_total"):
                    return sample.value
        return 0.0

    before = rate_limit_error_count()

    # Starlette's TestClient always reports "testclient" as the websocket
    # client host — saturate the limiter for that exact key so the real
    # route rejects the next connection attempt itself.
    for _ in range(_session_rate_limiter._max_requests):
        _session_rate_limiter.allow("testclient")

    client = TestClient(gateway_app)
    try:
        with client.websocket_connect("/ws/stt"):
            pass
    except Exception:
        pass  # a 1008 policy-violation close surfaces as a client-side error here — expected

    assert rate_limit_error_count() == before + 1


def test_tts_ws_is_also_rate_limited():
    """Real gap caught in a pre-deploy sweep: /ws/tts was the only one of
    the three voice-cost websockets that accepted unconditionally, with no
    rate check at all -- a client could open unlimited connections and
    drive unbounded billed synthesis calls. Same shape as the /ws/stt test
    above, just proving tts_ws now shares the same guard."""
    from agent_core.observability.metrics import errors_total
    from agent_core.speech_gateway.main import _session_rate_limiter

    def rate_limit_error_count():
        for metric in errors_total.collect():
            for sample in metric.samples:
                if sample.labels.get("stage") == "rate_limit" and sample.name.endswith("_total"):
                    return sample.value
        return 0.0

    before = rate_limit_error_count()

    for _ in range(_session_rate_limiter._max_requests):
        _session_rate_limiter.allow("testclient")

    client = TestClient(gateway_app)
    try:
        with client.websocket_connect("/ws/tts"):
            pass
    except Exception:
        pass  # a 1008 policy-violation close surfaces as a client-side error here — expected

    assert rate_limit_error_count() == before + 1
