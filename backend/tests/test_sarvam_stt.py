import base64
import json

import httpx
import pytest
import websockets.exceptions

import agent_core.speech.sarvam_stt as sarvam_stt_module
from agent_core.speech.sarvam_stt import SarvamSTTClient, SpeechStreamError
from agent_core.speech.clients import STTMode

from ._fake_ws import FakeWSConnection, fake_connect_returning


@pytest.fixture(autouse=True)
def _fast_post_audio_timeout(monkeypatch):
    """The real post-audio wait for Sarvam's final transcript is 5s — with the
    fake connection (which goes quiet after its scripted events, like a real
    quiet socket) every stream test would pay that in full. Shrunk here; the
    timeout mechanism itself is what the tests exercise, not its duration."""
    monkeypatch.setattr(sarvam_stt_module, "_POST_AUDIO_FINAL_EVENTS_TIMEOUT_SECONDS", 0.05)


async def _audio_frames(*frames):
    for f in frames:
        yield f


async def test_stream_translates_real_sarvam_events_and_sends_config_via_url_and_audio_as_base64_json(monkeypatch):
    """Regression test for the real bug found in production testing: the
    original protocol sent a post-connect `config` JSON message, raw binary
    audio frames, and expected `{"type":"transcript",...}`/`{"type":"vad",
    ...}` shaped events directly from Sarvam. Real Sarvam rejected audio
    frames with `'audio' must not be None` — the actual API takes config as
    URL query params, audio as base64-JSON, and returns
    `{"type":"data"/"events",...}` shaped events that this client must
    translate."""
    monkeypatch.setenv("SARVAM_API_KEY", "test-key")
    incoming = [
        json.dumps({"type": "events", "data": {"signal_type": "START_SPEECH"}}),
        json.dumps({"type": "data", "data": {"transcript": "hello", "language_probability": 0.92}}),
    ]
    ws = FakeWSConnection(incoming=incoming)
    connect_calls = []

    def connect(url, **kwargs):
        connect_calls.append((url, kwargs))
        return ws

    client = SarvamSTTClient(connect=connect)

    events = [
        e
        async for e in client.stream(
            _audio_frames(b"\x00\x01" * 512, b"\x00\x01" * 512), codec="pcm_s16le", sample_rate=16000
        )
    ]

    assert events == [
        {"type": "vad", "signal": "speech_start"},
        {"type": "transcript", "text": "hello", "is_final": True, "confidence": 0.92},
    ]

    # Config travels as URL query params, not a post-connect message.
    url, kwargs = connect_calls[0]
    assert "vad_signals=true" in url
    assert "high_vad_sensitivity=true" in url
    assert "mode=codemix" in url
    assert "input_audio_codec=pcm_s16le" in url
    # Real auth header, not Authorization: Bearer.
    assert kwargs["additional_headers"] == {"Api-Subscription-Key": "test-key"}

    # Audio was sent as base64-encoded JSON, never raw bytes.
    assert not any(isinstance(m, (bytes, bytearray)) for m in ws.sent)
    audio_messages = [json.loads(m) for m in ws.sent]
    assert audio_messages[0]["audio"]["data"] == base64.b64encode(b"\x00\x01" * 512).decode()
    assert audio_messages[0]["audio"]["sample_rate"] == "16000"


async def test_error_event_is_translated():
    error_event = json.dumps({"type": "error", "data": {"error": "Invalid request", "code": "bad_request"}})
    translated = SarvamSTTClient._translate_event(json.loads(error_event))
    assert translated == {"type": "error", "reason": "Invalid request"}


async def test_end_speech_vad_signal_is_translated():
    event = {"type": "events", "data": {"signal_type": "END_SPEECH"}}
    assert SarvamSTTClient._translate_event(event) == {"type": "vad", "signal": "speech_end"}


async def test_connection_closed_raises_speech_stream_error(monkeypatch):
    monkeypatch.setenv("SARVAM_API_KEY", "test-key")
    ws = FakeWSConnection(incoming=[], fail_with=websockets.exceptions.ConnectionClosed(None, None))
    client = SarvamSTTClient(connect=fake_connect_returning(ws))

    with pytest.raises(SpeechStreamError):
        async for _ in client.stream(_audio_frames(b"\x00\x00"), codec="pcm_s16le"):
            pass


async def test_missing_api_key_raises_before_connecting(monkeypatch):
    monkeypatch.delenv("SARVAM_API_KEY", raising=False)
    ws = FakeWSConnection(incoming=[])
    client = SarvamSTTClient(connect=fake_connect_returning(ws))

    with pytest.raises(SpeechStreamError):
        async for _ in client.stream(_audio_frames(b"\x00\x00"), codec="pcm_s16le"):
            pass


async def test_transcribe_rest_posts_audio_and_returns_json(monkeypatch):
    """Sarvam's REST response is translated into the SAME stable shape as the
    streaming path (`type`/`is_final`/`text`) — not passed through raw. A
    caller checking `is_final` (every real caller does) must see this
    fallback result as final, same as it would a streaming transcript."""
    monkeypatch.setenv("SARVAM_API_KEY", "test-key")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"transcript": "fallback transcript", "language_probability": 0.7})

    client = SarvamSTTClient(http_transport=httpx.MockTransport(handler))

    result = await client.transcribe_rest(b"\x00\x00\x00\x00")

    assert result == {"type": "transcript", "text": "fallback transcript", "is_final": True, "confidence": 0.7}


async def test_transcribe_rest_error_raises_speech_stream_error(monkeypatch):
    monkeypatch.setenv("SARVAM_API_KEY", "test-key")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="server error")

    client = SarvamSTTClient(http_transport=httpx.MockTransport(handler))

    with pytest.raises(SpeechStreamError):
        await client.transcribe_rest(b"\x00\x00")
