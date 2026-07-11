"""PII masking for transcripts before they reach logs/analytics (S2S plan §6).

Regex-based, not an ML PII detector — deliberately conservative and
zero-dependency rather than exhaustive NER. Longest/most-specific patterns
run first so a 12-digit ID number isn't first chewed up by the 10-digit
phone pattern. Extend the pattern list if a specific PII class proves
under-caught; this isn't meant to be a complete PII classifier.
"""

from __future__ import annotations

import re

_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "[EMAIL]"),
    (re.compile(r"\b(?:\d[ -]?){13,16}\b"), "[CARD_NUMBER]"),  # card-shaped runs, longest first
    (re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b"), "[ID_NUMBER]"),  # Aadhaar-shaped 12-digit
    (re.compile(r"\b[+]?\d{1,3}[-\s]?\d{10}\b"), "[PHONE]"),
    (re.compile(r"\b\d{10}\b"), "[PHONE]"),
]


def mask_pii(text: str) -> str:
    masked = text
    for pattern, replacement in _PATTERNS:
        masked = pattern.sub(replacement, masked)
    return masked
