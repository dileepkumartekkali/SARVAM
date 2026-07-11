from agent_core.speech.audio_validation import validate_pcm_frame, validate_wav


def test_valid_wav_header_accepted():
    wav_bytes = b"RIFF" + (100).to_bytes(4, "little") + b"WAVE" + b"\x00" * 40
    assert validate_wav(wav_bytes).ok is True


def test_missing_riff_magic_rejected():
    fake = b"NOTWAVDATA" + b"\x00" * 40
    result = validate_wav(fake)
    assert result.ok is False
    assert "magic" in result.reason


def test_extension_alone_is_never_trusted():
    """A .wav-looking-but-not-actually-WAV payload must be rejected regardless
    of what a client claims about it — only the bytes matter."""
    fake_wav_bytes = b"just some plain text pretending to be audio"
    assert validate_wav(fake_wav_bytes).ok is False


def test_valid_pcm16_frame_at_16khz_accepted():
    expected_frame_bytes = int(16000 * 0.032) * 2  # 32ms @ 16kHz, 16-bit
    frame = b"\x00\x01" * (expected_frame_bytes // 2)
    assert validate_pcm_frame(frame, sample_rate=16000).ok is True


def test_odd_byte_count_rejected_as_truncated_sample():
    frame = b"\x00\x01\x02"  # 3 bytes — not a whole number of 16-bit samples
    result = validate_pcm_frame(frame, sample_rate=16000)
    assert result.ok is False
    assert "16-bit" in result.reason


def test_unsupported_sample_rate_rejected():
    result = validate_pcm_frame(b"\x00\x00", sample_rate=44100)
    assert result.ok is False
    assert "sample rate" in result.reason


def test_empty_frame_rejected():
    assert validate_pcm_frame(b"", sample_rate=16000).ok is False


def test_oversized_frame_rejected():
    huge = b"\x00\x00" * 100_000
    result = validate_pcm_frame(huge, sample_rate=16000)
    assert result.ok is False
