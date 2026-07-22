"""Fake STT/TTS clients used ONLY when MAAV_LOAD_TEST_MODE=true.

Load testing 100-10,000 concurrent voice sessions against a real Sarvam
account isn't possible in this environment (no credentials, and it would
also be irresponsible to hammer a real paid third-party API for a load
test). This module simulates minimal realistic latency (a few ms) so a load
test measures OUR gateway's own connection-handling capacity — the
process/asyncio/OS-level ceiling — which is one of the three candidate
bottlenecks the load-test report evaluates (the other two: backend CPU,
Sarvam's own undocumented rate ceiling, which cannot be measured here at all
and is called out as such in the report).
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator

from ..speech.clients import STTEvent, STTMode


class LoadTestFakeSTT:
    name = "load-test-fake-stt"

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
        frame_count = 0
        async for _ in audio:
            frame_count += 1
            await asyncio.sleep(0.005)  # simulate minimal real processing latency
            if frame_count == 3:
                yield {"type": "transcript", "text": "load test transcript", "is_final": True, "confidence": 0.95}

    async def transcribe_rest(self, audio: bytes, *, mode: STTMode = STTMode.CODEMIX) -> STTEvent:
        return {"text": "load test rest transcript"}

    async def transcribe_batch(self, audio_uri: str, *, mode: STTMode = STTMode.CODEMIX) -> str:
        return "load-test-job"


class LoadTestFakeTTS:
    async def synthesize(self, text_chunks, *, language, model="bulbul:v2", voice=None, pace=None):
        async for _ in text_chunks:
            await asyncio.sleep(0.005)
            yield b"\x00\x00" * 100
