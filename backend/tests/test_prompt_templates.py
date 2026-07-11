"""_build_system_prompt reads real template files and substitutes variables —
never an inline string, and logs only a version id, never the full text."""

from agent_core.agents.task_agent import _build_system_prompt
from agent_core.supervisor.state import Mode, SessionState


def test_text_mode_loads_text_template_and_substitutes_vars():
    session = SessionState(
        session_id="s", conversation_id="c", thread_id="t", mode=Mode.TEXT_TO_TEXT,
        response_language="hi", language_confidence=0.87, is_code_mixed=True,
    )

    text, version_id = _build_system_prompt(session)

    assert version_id == "text_mode_system.v1"
    assert "hi" in text
    assert "0.87" in text
    assert "True" in text
    assert "Markdown (lists, headers, bold) is permitted" in text  # TEXT_MODE-only line


def test_voice_mode_loads_voice_template():
    session = SessionState(
        session_id="s", conversation_id="c", thread_id="t", mode=Mode.SPEECH_TO_SPEECH,
        response_language="ta", language_confidence=0.5, is_code_mixed=False,
    )

    text, version_id = _build_system_prompt(session)

    assert version_id == "voice_mode_system.v1"
    assert "No markdown" in text  # VOICE_MODE-only line
    assert "ta" in text


def test_missing_language_context_renders_unknown():
    session = SessionState(session_id="s", conversation_id="c", thread_id="t")

    text, _ = _build_system_prompt(session)

    assert "unknown" in text
