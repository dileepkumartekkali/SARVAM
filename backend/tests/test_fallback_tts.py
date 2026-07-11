"""Azure fallback TTS client — httpx.MockTransport, no network, no real key."""

import httpx

from agent_core.speech.fallback_tts import AzureFallbackTTSClient


async def _texts(*parts):
    for p in parts:
        yield p


async def test_synthesize_posts_ssml_with_correct_voice_and_returns_audio(monkeypatch):
    monkeypatch.setenv("AZURE_SPEECH_REGION", "centralindia")
    monkeypatch.setenv("AZURE_SPEECH_KEY", "test-key")

    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["headers"] = request.headers
        captured["body"] = request.content.decode("utf-8")
        return httpx.Response(200, content=b"FAKE_AUDIO_BYTES")

    transport = httpx.MockTransport(handler)
    client = AzureFallbackTTSClient(transport=transport)

    chunks = [c async for c in client.synthesize(_texts("Hello there"), language="as")]

    assert chunks == [b"FAKE_AUDIO_BYTES"]
    assert "centralindia.tts.speech.microsoft.com" in captured["url"]
    assert captured["headers"]["ocp-apim-subscription-key"] == "test-key"
    assert "as-IN-YashicaNeural" in captured["body"]
    assert "Hello there" in captured["body"]


async def test_urdu_uses_urdu_voice(monkeypatch):
    monkeypatch.setenv("AZURE_SPEECH_REGION", "centralindia")
    monkeypatch.setenv("AZURE_SPEECH_KEY", "test-key")
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = request.content.decode("utf-8")
        return httpx.Response(200, content=b"AUDIO")

    client = AzureFallbackTTSClient(transport=httpx.MockTransport(handler))
    _ = [c async for c in client.synthesize(_texts("hello"), language="ur")]

    assert "ur-IN-GulNeural" in captured["body"]


async def test_missing_key_raises_clear_error(monkeypatch):
    monkeypatch.delenv("AZURE_SPEECH_KEY", raising=False)
    monkeypatch.setenv("AZURE_SPEECH_REGION", "centralindia")
    client = AzureFallbackTTSClient()

    try:
        _ = [c async for c in client.synthesize(_texts("hello"), language="as")]
        assert False, "expected RuntimeError"
    except RuntimeError as e:
        assert "AZURE_SPEECH_KEY" in str(e)
