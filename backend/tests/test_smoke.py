"""Smoke test: the package imports and the one real branch (Mode.is_voice) holds.

Everything else this phase is a typed stub with no behavior to test yet.
"""

from agent_core.api.main import app
from agent_core.llm_adapter import LLMProvider, LLMRouter  # importable
from agent_core.speech import SpeechSTTClient, SpeechTTSClient  # importable
from agent_core.supervisor import Mode, SessionState


def test_app_boots():
    assert any(r.path == "/health" for r in app.routes)


def test_voice_mode_selection():
    assert Mode.SPEECH_TO_SPEECH.is_voice
    assert Mode.TEXT_TO_SPEECH.is_voice
    assert not Mode.TEXT_TO_TEXT.is_voice
    assert not Mode.SPEECH_TO_TEXT.is_voice


def test_session_state_defaults():
    s = SessionState(session_id="s", conversation_id="c", thread_id="t")
    assert s.mode is Mode.TEXT_TO_TEXT
    assert s.thread_id == "t"  # checkpointer key
