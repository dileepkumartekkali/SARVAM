"""Gateway-side audio validation (S2S plan §5: "Malformed/oversized audio
upload... Gateway-side validation before touching Sarvam").

The client (mic-capture code) is contractually required to encode PCM16/WAV
before it ever opens the WebSocket (S2S plan §2) — but a client claim isn't a
guarantee, and a malicious or buggy client could send anything. This module is
the actual enforced boundary: magic-byte/frame-size checks on the bytes the
gateway received, run before a single byte is forwarded to Sarvam (the
"validation chokepoint" reason for having a gateway at all). Extension or
Content-Type headers are never trusted — only the bytes.
"""

from __future__ import annotations

from dataclasses import dataclass

# 16-bit samples, mono, at these two rates only (S2S plan §1: 16kHz standard,
# 8kHz telephony now supported).
_ALLOWED_SAMPLE_RATES = (8000, 16000)
_BYTES_PER_SAMPLE = 2  # PCM16
_FRAME_DURATION_SECONDS = 0.032  # 32ms frames per the S2S plan

MAX_FRAME_BYTES = 64_000  # generous upper bound per single frame — cost/DoS guard


@dataclass
class AudioValidationResult:
    ok: bool
    reason: str = ""


def _is_wav_header(data: bytes) -> bool:
    return len(data) >= 12 and data[0:4] == b"RIFF" and data[8:12] == b"WAVE"


def validate_wav(data: bytes) -> AudioValidationResult:
    if not _is_wav_header(data):
        return AudioValidationResult(False, "not a valid WAV file (missing RIFF/WAVE magic bytes)")
    if len(data) > MAX_FRAME_BYTES * 100:  # a full WAV upload, not a single frame — looser cap
        return AudioValidationResult(False, "WAV payload exceeds the maximum allowed size")
    return AudioValidationResult(True)


def validate_pcm_frame(data: bytes, *, sample_rate: int) -> AudioValidationResult:
    """Validates a single raw PCM16 frame: correct sample rate, even byte
    count (whole 16-bit samples, never a truncated sample), and a size that's
    plausible for a 32ms frame at the claimed rate — catches both garbage and
    oversized/DoS-shaped payloads.
    """
    if sample_rate not in _ALLOWED_SAMPLE_RATES:
        return AudioValidationResult(False, f"unsupported sample rate: {sample_rate}")
    if len(data) == 0:
        return AudioValidationResult(False, "empty frame")
    if len(data) % _BYTES_PER_SAMPLE != 0:
        return AudioValidationResult(False, "frame byte count is not a whole number of 16-bit samples")
    if len(data) > MAX_FRAME_BYTES:
        return AudioValidationResult(False, "frame exceeds maximum allowed size")

    expected_frame_bytes = int(sample_rate * _FRAME_DURATION_SECONDS) * _BYTES_PER_SAMPLE
    # Allow a generous band around the nominal 32ms frame size — clients may
    # batch a few frames together; only reject payloads wildly outside that.
    if len(data) > expected_frame_bytes * 20:
        return AudioValidationResult(False, "frame far exceeds the expected 32ms-frame size")

    return AudioValidationResult(True)
