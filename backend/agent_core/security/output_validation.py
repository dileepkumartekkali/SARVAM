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
