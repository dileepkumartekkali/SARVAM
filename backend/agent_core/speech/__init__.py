"""Sarvam STT/TTS clients, the response chunker, and audio/language policy.

The Speech Gateway (`agent_core.speech_gateway`) is the service that holds
Sarvam keys and wires these pieces together — this package only defines the
clients and pure-logic pieces it depends on.
"""

from .audio_validation import AudioValidationResult, validate_pcm_frame, validate_wav
from .chunker import chunk_stream
from .clients import SpeechSTTClient, SpeechTTSClient, STTMode
from .fallback_tts import AzureFallbackTTSClient
from .sarvam_stt import SarvamSTTClient, SpeechStreamError
from .sarvam_tts import SarvamTTSClient, TTSStreamError
from .tts_provider_policy import NOT_SUPPORTED_BY_SARVAM_TTS, select_tts_provider

__all__ = [
    "SpeechSTTClient",
    "SpeechTTSClient",
    "STTMode",
    "SarvamSTTClient",
    "SpeechStreamError",
    "SarvamTTSClient",
    "TTSStreamError",
    "AzureFallbackTTSClient",
    "select_tts_provider",
    "NOT_SUPPORTED_BY_SARVAM_TTS",
    "chunk_stream",
    "validate_wav",
    "validate_pcm_frame",
    "AudioValidationResult",
]
