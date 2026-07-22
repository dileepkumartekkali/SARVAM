"""Azure Cognitive Services neural TTS — the fallback for languages Bulbul
doesn't support (see tts_provider_policy.py). Confirmed against Microsoft's
live docs (learn.microsoft.com/.../speech-service/language-support, fetched
July 2026) that Azure has real neural voices for both gap languages:

    as-IN-YashicaNeural / as-IN-PriyomNeural   (Assamese)
    ur-IN-GulNeural / ur-IN-SalmanNeural        (Urdu, India)
    ur-PK-UzmaNeural / ur-PK-AsadNeural          (Urdu, Pakistan)

REST, not streaming — Azure's SSML synthesis endpoint returns the full audio
in one response. It still implements `SpeechTTSClient.synthesize()` (an
AsyncIterator[bytes]) so the gateway can treat it interchangeably with the
Sarvam streaming client; it just yields the whole payload as a single chunk
instead of several, which slightly increases TTFB for this fallback path — an
acceptable tradeoff since it only serves two languages Sarvam can't.

Google Cloud TTS is the documented alternative and also covers both languages
— not implemented here to avoid maintaining two redundant fallback clients;
swap this module for Google's REST API if Azure isn't the preferred vendor.
"""

from __future__ import annotations

import os
from typing import AsyncIterator
from xml.sax.saxutils import escape

import httpx

_DEFAULT_VOICES = {
    "as": "as-IN-YashicaNeural",
    "ur": "ur-IN-GulNeural",
}


class AzureFallbackTTSClient:
    def __init__(
        self,
        *,
        region_env: str = "AZURE_SPEECH_REGION",
        api_key_env: str = "AZURE_SPEECH_KEY",
        voices: dict[str, str] | None = None,
        timeout: float = 30.0,
        transport: httpx.BaseTransport | None = None,
    ):
        self._region_env = region_env
        self._api_key_env = api_key_env
        self._voices = voices or _DEFAULT_VOICES
        self._timeout = timeout
        self._transport = transport  # test-injection hook only; None in prod

    def _region(self) -> str:
        region = os.environ.get(self._region_env)
        if not region:
            raise RuntimeError(f"{self._region_env} not set")
        return region

    def _api_key(self) -> str:
        key = os.environ.get(self._api_key_env)
        if not key:
            raise RuntimeError(f"{self._api_key_env} not set")
        return key

    def _voice_for(self, language: str) -> str:
        voice = self._voices.get(language)
        if voice is None:
            raise ValueError(f"no Azure fallback voice configured for language {language!r}")
        return voice

    async def synthesize(
        self,
        text_chunks: AsyncIterator[str],
        *,
        language: str,
        model: str = "bulbul:v2",  # ignored — Azure has its own voice models, kept for interface parity
        voice: str | None = None,
        pace: float | None = None,
    ) -> AsyncIterator[bytes]:
        voice_name = voice or self._voice_for(language)
        text = "".join([chunk async for chunk in text_chunks])
        rate_attr = f' rate="{pace}"' if pace is not None else ""
        ssml = (
            '<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US">'
            f'<voice name="{voice_name}"><prosody{rate_attr}>{escape(text)}</prosody></voice>'
            "</speak>"
        )
        url = f"https://{self._region()}.tts.speech.microsoft.com/cognitiveservices/v1"
        headers = {
            "Ocp-Apim-Subscription-Key": self._api_key(),
            "Content-Type": "application/ssml+xml",
            "X-Microsoft-OutputFormat": "riff-24khz-16bit-mono-pcm",
            "User-Agent": "maav-speech-gateway",
        }
        async with httpx.AsyncClient(timeout=self._timeout, transport=self._transport) as client:
            resp = await client.post(url, content=ssml.encode("utf-8"), headers=headers)
            if resp.status_code != 200:
                raise RuntimeError(f"Azure TTS returned {resp.status_code}: {resp.text}")
            yield resp.content
