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

_SCRIPT_TAG = re.compile(r"<\s*script\b.*?>.*?<\s*/\s*script\s*>", re.IGNORECASE | re.DOTALL)
_HTML_TAG = re.compile(r"<[^>]+>")

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
    return cleaned
