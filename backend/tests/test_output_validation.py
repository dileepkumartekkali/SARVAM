from agent_core.security.output_validation import marker_safe_split, sanitize_llm_output
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


def test_marker_safe_split_across_an_adversarial_chunk_boundary():
    """Real bug hit live, twice over: (1) checking each streamed chunk in
    isolation misses a marker split across two deltas, since neither half
    alone contains the full string. (2) The first fix attempt at this
    function searched for the marker only in the "safe" portion after
    already splitting -- if the marker straddled that split point, it got
    permanently cut in half (opening bracket stranded with no closing
    bracket, and vice versa) and could never be detected as whole again.
    This deliberately splits the marker AT ITS MIDPOINT, the worst case,
    to prove it's still fully removed once both pieces are fed through."""
    text = f"The CEO is Real Person.\n{TOOL_VERIFIED_MARKER}"
    midpoint = len(text) // 2  # guaranteed to land inside the marker itself
    first_delta, second_delta = text[:midpoint], text[midpoint:]

    tail = ""
    released = []
    for delta in (first_delta, second_delta):
        tail += delta
        safe, tail = marker_safe_split(tail)
        if safe:
            released.append(safe)
    released.append(tail)  # final flush, as stream_turn does once the stream ends

    combined = "".join(released)
    assert TOOL_VERIFIED_MARKER not in combined
    assert combined.strip() == "The CEO is Real Person."


def test_marker_safe_split_holds_back_a_forming_partial_marker():
    """A partial marker prefix at the very tail must never be released
    early, even though it doesn't match the full pattern yet -- releasing
    it would make it impossible to strip once the rest arrives in a LATER
    chunk."""
    safe, tail = marker_safe_split("Some real text [This reply was verif")

    assert "[This reply was verif" not in safe  # the forming prefix stays held back
    assert "[This reply was verif" in tail
