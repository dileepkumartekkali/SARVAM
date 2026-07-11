"""Session state threaded through the LangGraph supervisor.

This is data, not behavior, so it's a pydantic v2 model rather than a Protocol.
It is the single source of truth for the two things the rest of the system
branches on: which prompt variant to build (`mode` → VOICE vs TEXT) and which
language to answer in (`response_language` + confidence).
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class Mode(str, Enum):
    """Pipeline mode for a turn. Determines the prompt variant.

    VOICE_MODE (stricter: 1–3 sentences, no markdown/emoji) applies whenever the
    final output goes through TTS — i.e. TEXT_TO_SPEECH and SPEECH_TO_SPEECH.
    Everything else is TEXT_MODE.
    """

    TEXT_TO_TEXT = "text_to_text"
    SPEECH_TO_TEXT = "speech_to_text"
    TEXT_TO_SPEECH = "text_to_speech"
    SPEECH_TO_SPEECH = "speech_to_speech"

    @property
    def is_voice(self) -> bool:
        """True when output is spoken → use the stricter VOICE_MODE prompt."""
        return self in (Mode.TEXT_TO_SPEECH, Mode.SPEECH_TO_SPEECH)


class SessionState(BaseModel):
    """State for one turn / session, checkpointed per `thread_id`.

    `thread_id` is the LangGraph checkpointer key: the client reconnects with
    the same value to resume an in-progress turn instead of resetting.
    """

    # Identity / checkpointer key
    session_id: str
    conversation_id: str
    thread_id: str = Field(description="LangGraph checkpointer key; stable across reconnects")

    # Mode → prompt variant
    mode: Mode = Mode.TEXT_TO_TEXT

    # Language context (fed into the system prompt template variables)
    response_language: str | None = None
    language_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    is_code_mixed: bool = False
    translation_applied: bool = False

    # Data retention (S2S plan §6): raw audio is ephemeral by default — buffer,
    # transcribe, discard. Only an explicit consent flag on the session
    # permits persisting it (e.g. for QA/training); nothing in this codebase
    # currently persists raw audio, so this is the gate any future code that
    # wants to must check first.
    audio_retention_consent: bool = False

    model_config = {"use_enum_values": False}
