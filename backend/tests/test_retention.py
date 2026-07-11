import pytest

from agent_core.security.retention import RetentionNotConsented, assert_audio_persistence_allowed
from agent_core.supervisor.state import SessionState


def _session(**kwargs) -> SessionState:
    return SessionState(session_id="s", conversation_id="c", thread_id="t", **kwargs)


def test_persistence_blocked_by_default():
    with pytest.raises(RetentionNotConsented):
        assert_audio_persistence_allowed(_session())


def test_persistence_allowed_with_explicit_consent():
    assert_audio_persistence_allowed(_session(audio_retention_consent=True))  # does not raise
