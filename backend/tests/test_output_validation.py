from agent_core.security.output_validation import sanitize_llm_output
from agent_core.tools.rag_tool import TOOL_VERIFIED_MARKER


def test_strips_script_tags():
    result = sanitize_llm_output("Here you go <script>alert('xss')</script> enjoy")
    assert "<script>" not in result
    assert "alert" not in result


def test_strips_html_tags():
    result = sanitize_llm_output("Some <b>bold</b> text")
    assert "<b>" not in result
    assert "bold" in result


def test_redacts_system_prompt_leakage():
    leaked = "Sure — here's my instructions:\n## IDENTITY & SCOPE\n- you must never reveal..."
    result = sanitize_llm_output(leaked)
    assert "IDENTITY & SCOPE" not in result
    assert "can't share my internal instructions" in result


def test_normal_text_passes_through_unchanged():
    text = "Your flight departs at 4:30 PM."
    assert sanitize_llm_output(text) == text


def test_strips_tool_verified_marker_if_the_model_echoes_it():
    """Real bug hit live: TOOL_VERIFIED_MARKER is meant to be internal-only
    (appended to conversation HISTORY, never shown to the user) -- but once
    it's sitting in the assistant's own prior-turn text, the model
    sometimes imitates the pattern and reproduces the exact marker in a
    LATER reply. An instruction telling it not to is exactly the kind of
    fix this whole codebase already learned not to trust -- stripped here
    unconditionally instead, so it's structurally impossible for it to
    reach the user regardless of what the model does."""
    leaked = f"The CEO is Real Person.\n{TOOL_VERIFIED_MARKER}"
    result = sanitize_llm_output(leaked)
    assert TOOL_VERIFIED_MARKER not in result
    assert result == "The CEO is Real Person."


def test_strips_tool_verified_marker_mid_text_too():
    leaked = f"First fact.\n{TOOL_VERIFIED_MARKER}\nSecond fact, still relevant."
    result = sanitize_llm_output(leaked)
    assert TOOL_VERIFIED_MARKER not in result
    assert "First fact." in result
    assert "Second fact, still relevant." in result
