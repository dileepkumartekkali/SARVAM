"""Output validation on LLM responses before they reach the client (OWASP
pass, S2S plan §6) — defense-in-depth against a prompt-injected or
malfunctioning model emitting content that becomes an XSS payload if the
frontend ever renders it as HTML, or that leaks the system prompt verbatim.

This does not replace the frontend's own output-encoding responsibility
(React escapes by default) — it's a second layer so a compromise of that
assumption doesn't immediately become exploitable.
"""

from __future__ import annotations

import re

from ..tools.rag_tool import TOOL_VERIFIED_MARKER

_SCRIPT_TAG = re.compile(r"<\s*script\b.*?>.*?<\s*/\s*script\s*>", re.IGNORECASE | re.DOTALL)
_HTML_TAG = re.compile(r"<[^>]+>")
# Real bug hit live: TOOL_VERIFIED_MARKER is meant to be internal-only,
# appended to conversation HISTORY so a future turn can tell a real answer
# from a guess (see rag_tool.py) -- never shown to the user. But once it
# sits in the assistant's own prior-turn text, the model sometimes imitates
# the pattern and reproduces the exact marker text in a LATER reply, since
# it looks like part of "how this conversation's assistant turns are
# styled." Relying on an instruction not to do this is exactly the kind of
# thing this whole codebase already learned not to trust (see task_agent.py's
# own history of prompt-only fixes proving unreliable) -- stripped here
# unconditionally instead, so it is structurally impossible for it to reach
# the user regardless of what the model does.
_TOOL_VERIFIED_MARKER_RE = re.compile(re.escape(TOOL_VERIFIED_MARKER))

# Distinctive section headers from the real system prompt — their presence
# in model output means the prompt leaked, not that the user asked about them.
_SYSTEM_PROMPT_LEAK_MARKERS = (
    "IDENTITY & SCOPE",
    "LANGUAGE PRESERVATION (non-negotiable)",
)

_LEAK_REFUSAL = "I can't share my internal instructions, but I'm happy to help with your question."


def sanitize_llm_output(text: str) -> str:
    if any(marker in text for marker in _SYSTEM_PROMPT_LEAK_MARKERS):
        return _LEAK_REFUSAL
    cleaned = _SCRIPT_TAG.sub("", text)
    cleaned = _HTML_TAG.sub("", cleaned)
    cleaned = _TOOL_VERIFIED_MARKER_RE.sub("", cleaned)
    # Collapse the blank line(s) the removed marker leaves behind, and any
    # trailing/leading whitespace it exposes at the very ends of the reply.
    cleaned = re.sub(r"\n{2,}", "\n", cleaned).strip()
    return cleaned


def strip_tool_verified_marker(text: str) -> str:
    """Lighter than sanitize_llm_output() -- only strips the exact marker
    text, never the full-reply HTML/system-prompt-leak checks above. Safe
    for a single already-complete piece of text; NOT safe alone for a
    stream of arbitrarily-split chunks -- see marker_safe_split() below for
    why a per-chunk-only check was proven insufficient."""
    return _TOOL_VERIFIED_MARKER_RE.sub("", text)


_MARKER_LEN = len(TOOL_VERIFIED_MARKER)


_LEAK_MARKER_MAX_LEN = max(len(m) for m in _SYSTEM_PROMPT_LEAK_MARKERS)


def stream_safe_sanitize(buffered: str) -> tuple[str, str, bool]:
    """Real gap hit live: `stream_turn` only ever ran `marker_safe_split`
    (TOOL_VERIFIED_MARKER only) on live `text_delta` chunks -- the
    system-prompt-leak refusal and script/HTML stripping in
    `sanitize_llm_output` only ran on the terminal `done` event, whose text
    is DISCARDED on the success path (the frontend already rendered the raw
    deltas). A prompt-injected/malfunctioning model emitting the system
    prompt or a `<script>` payload reached the client verbatim through live
    deltas -- the leak-refusal and script/HTML defenses were fully inert on
    the primary streaming path.

    Same holdback philosophy as `marker_safe_split`, generalized:
    - TOOL_VERIFIED_MARKER and the leak markers are fixed-length strings --
      hold back enough of the tail that a partial occurrence forming right
      at the boundary can never be released prematurely, exactly like
      `marker_safe_split` already does for the one marker.
    - HTML/script tags have no fixed length, so instead of a length-based
      holdback, anything from an unclosed trailing `<` onward is held back
      until its matching `>` arrives (a `<script>` block of any length is
      still fully removed once closed -- it just isn't released piecemeal
      while open).

    Returns (safe_to_yield, new_tail_to_keep_buffering, leak_detected).
    Once a full system-prompt-leak marker is found, `leak_detected` is True
    and `safe_to_yield` is the leak refusal -- the caller must stop
    streaming any further real content for this turn (already-yielded
    chunks before detection can't be recalled over SSE; this stops the
    rest of the leak from continuing to stream, which is the same
    trade-off streaming turns already accept elsewhere in this module for
    self-check correction).
    """
    cleaned = _SCRIPT_TAG.sub("", buffered)
    cleaned = _HTML_TAG.sub("", cleaned)
    cleaned = _TOOL_VERIFIED_MARKER_RE.sub("", cleaned)

    if any(marker in cleaned for marker in _SYSTEM_PROMPT_LEAK_MARKERS):
        return _LEAK_REFUSAL, "", True

    hold_back = _MARKER_LEN - 1
    hold_back = max(hold_back, _LEAK_MARKER_MAX_LEN - 1)
    last_open = cleaned.rfind("<")
    if last_open != -1 and cleaned.find(">", last_open) == -1:
        # An unclosed tag/script-open is forming at the tail -- hold back
        # everything from it onward regardless of the fixed-marker amount.
        hold_back = max(hold_back, len(cleaned) - last_open)

    if len(cleaned) <= hold_back:
        return "", cleaned, False
    split_at = len(cleaned) - hold_back
    return cleaned[:split_at], cleaned[split_at:], False


def marker_safe_split(buffered: str) -> tuple[str, str]:
    """Real gap caught by this fix's OWN regression test, twice over: (1) a
    per-chunk-only strip isn't enough -- the chunker can split the marker
    text itself across two separate deltas, so neither individual chunk
    contains the full string and checking each one in isolation misses it
    entirely. (2) The first attempt at THIS function searched for the
    marker only within the "safe" portion after splitting, not the whole
    buffer -- if the marker straddled the split point, it could be cut
    exactly in half by the split itself (its opening stranded in "safe"
    with no closing bracket, its closing stranded in "tail" with no
    opening bracket), so NEITHER half would ever match the full pattern
    and it would leak forever, never becoming whole again.

    Correct order: strip the FULL marker from the ENTIRE buffered text
    FIRST (a complete occurrence, wherever it started, is always found
    this way), and only THEN decide the safe/tail split on what's left.
    Holds back the last (marker_len - 1) characters of the CLEANED text --
    a partial marker still forming can never be longer than that, so it's
    always fully contained in what's held back, never in what's released.
    Returns (safe_to_yield, new_tail_to_keep_buffering)."""
    cleaned = _TOOL_VERIFIED_MARKER_RE.sub("", buffered)
    if len(cleaned) <= _MARKER_LEN - 1:
        return "", cleaned
    split_at = len(cleaned) - (_MARKER_LEN - 1)
    return cleaned[:split_at], cleaned[split_at:]
