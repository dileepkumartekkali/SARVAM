from agent_core.security.output_validation import sanitize_llm_output


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
