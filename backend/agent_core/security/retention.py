"""Data retention gate (S2S plan §6): raw audio is ephemeral by default —
buffer, transcribe, discard. Nothing in this codebase persists raw audio
today; this function is the gate any future code path that wants to must call
first, so "persist audio" can never happen silently without a session's
explicit consent flag.
"""

from __future__ import annotations

from ..supervisor.state import SessionState


class RetentionNotConsented(Exception):
    """Raised when code attempts to persist audio for a session that never
    opted in."""


def assert_audio_persistence_allowed(session: SessionState) -> None:
    if not session.audio_retention_consent:
        raise RetentionNotConsented(
            f"session {session.session_id} has not consented to audio retention — "
            "persisting its audio is not allowed"
        )
