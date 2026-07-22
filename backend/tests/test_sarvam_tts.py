import base64
import json

import pytest

from agent_core.speech.sarvam_tts import SarvamTTSClient

from ._fake_ws import FakeWSConnection, fake_connect_returning


async def _texts(*parts):
    for p in parts:
        yield p


def _audio_event(raw: bytes) -> str:
    return json.dumps({"type": "audio", "data": {"audio": base64.b64encode(raw).decode(), "content_type": "audio/mp3"}})


_FINAL_EVENT = json.dumps({"type": "event", "data": {"event_type": "final", "message": "Processing completed"}})


async def test_synthesize_reuses_one_socket_for_all_chunks(monkeypatch):
    """One `connect()` call for the whole utterance — not one per chunk."""
    monkeypatch.setenv("SARVAM_API_KEY", "test-key")
    connect_calls = []

    incoming = [
        _audio_event(b"AUDIO_CHUNK_1"),
        _FINAL_EVENT,
        _audio_event(b"AUDIO_CHUNK_2"),
        _FINAL_EVENT,
    ]
    ws = FakeWSConnection(incoming=incoming)

    def connect(url, **kwargs):
        connect_calls.append(url)
        return ws

    client = SarvamTTSClient(connect=connect)

    audio = [a async for a in client.synthesize(_texts("Hello", "world"), language="hi")]

    assert audio == [b"AUDIO_CHUNK_1", b"AUDIO_CHUNK_2"]
    assert len(connect_calls) == 1  # exactly one socket for the whole utterance
    # Two text messages went out over that same socket, per the real protocol.
    text_messages = [json.loads(m) for m in ws.sent if isinstance(m, str) and '"text"' in m and '"config"' not in m]
    assert len(text_messages) == 2
    assert text_messages[0]["type"] == "text"
    assert text_messages[0]["data"]["text"] == "Hello"


async def test_config_message_uses_real_sarvam_field_names(monkeypatch):
    """Regression test for the real bug found in production testing: the
    original protocol used flat `language`/`voice` fields and a `convert`
    type — Sarvam rejected it with a 422. The real API needs a nested `data`
    object with `target_language_code` and a required `speaker`."""
    monkeypatch.setenv("SARVAM_API_KEY", "test-key")
    ws = FakeWSConnection(incoming=[_FINAL_EVENT])
    client = SarvamTTSClient(connect=fake_connect_returning(ws))

    _ = [a async for a in client.synthesize(_texts("hi"), language="hi", model="bulbul:v3")]

    config_message = json.loads(ws.sent[0])
    assert config_message["type"] == "config"
    assert config_message["data"]["target_language_code"] == "hi-IN"
    assert "speaker" in config_message["data"]  # required by Sarvam — must never be omitted
    assert "language" not in config_message  # the old, wrong, flat field


async def test_config_requests_raw_pcm_not_the_mp3_default(monkeypatch):
    """Real bug hit live: reported as "TTS speaking not clear" -- Sarvam
    defaults output_audio_codec to MP3 when omitted (confirmed against
    their real docs), and this config never set it. Each streamed chunk is
    a fragment of a continuous MP3 stream, not a self-contained file --
    decoding arbitrary fragments in isolation (this app's whole
    chunk-by-chunk playback model) produces glitchy audio. Requesting
    linear16 (raw PCM) instead means every chunk decodes cleanly on its
    own, matching how it's actually played."""
    monkeypatch.setenv("SARVAM_API_KEY", "test-key")
    ws = FakeWSConnection(incoming=[_FINAL_EVENT])
    client = SarvamTTSClient(connect=fake_connect_returning(ws))

    _ = [a async for a in client.synthesize(_texts("hi"), language="hi", model="bulbul:v3")]

    config_message = json.loads(ws.sent[0])
    assert config_message["data"]["output_audio_codec"] == "linear16"
    # bulbul:v3's own documented default -- explicit, not left to guesswork
    # on either side of the connection.
    assert config_message["data"]["speech_sample_rate"] == 24000


async def test_error_event_raises_tts_stream_error(monkeypatch):
    monkeypatch.setenv("SARVAM_API_KEY", "test-key")
    error_event = json.dumps({"type": "error", "data": {"message": "bad request", "code": 422}})
    ws = FakeWSConnection(incoming=[error_event])
    client = SarvamTTSClient(connect=fake_connect_returning(ws))

    from agent_core.speech.sarvam_tts import TTSStreamError

    with pytest.raises(TTSStreamError):
        _ = [a async for a in client.synthesize(_texts("hi"), language="hi")]


async def test_v3_uses_temperature_not_pitch_loudness():
    data = SarvamTTSClient._config_data("hi", "bulbul:v3", voice=None, pace=1.0, pitch=5, loudness=5, temperature=0.7)

    assert "temperature" in data
    assert "pitch" not in data
    assert "loudness" not in data


async def test_v2_uses_pitch_loudness_not_temperature():
    data = SarvamTTSClient._config_data("hi", "bulbul:v2", voice=None, pace=1.0, pitch=5, loudness=5, temperature=0.7)

    assert "pitch" in data
    assert "loudness" in data
    assert "temperature" not in data


async def test_pace_out_of_range_for_model_rejected():
    with pytest.raises(ValueError):
        SarvamTTSClient._config_data("hi", "bulbul:v3", voice=None, pace=2.5)  # v3 max is 2.0


async def test_default_speaker_matches_the_selected_model():
    """Real bug hit live, reported as "English is clear, Telugu is not":
    the old single flat default speaker ("anushka") is bulbul:v2-only, but
    every live call defaults to model="bulbul:v3" -- an invalid speaker/
    model pair per Sarvam's own docs ("speaker selection must match the
    chosen model version"). Each model must get a speaker that's actually
    valid FOR that model when no explicit voice is requested."""
    v3_data = SarvamTTSClient._config_data("hi", "bulbul:v3", voice=None, pace=None)
    assert v3_data["speaker"] == "shubh"

    v2_data = SarvamTTSClient._config_data("hi", "bulbul:v2", voice=None, pace=None)
    assert v2_data["speaker"] == "anushka"

    # An explicitly requested voice always wins over either default.
    explicit = SarvamTTSClient._config_data("hi", "bulbul:v3", voice="priya", pace=None)
    assert explicit["speaker"] == "priya"

    # Same pace is valid for v2 (range 0.3-3.0).
    SarvamTTSClient._config_data("hi", "bulbul:v2", voice=None, pace=2.5)
