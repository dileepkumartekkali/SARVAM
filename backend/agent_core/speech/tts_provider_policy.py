"""TTS provider selection per language — resolves the Assamese/Urdu gap that
speech_to_speech_plan.md §1 flagged as "unverified... needs a fallback
provider if required."

Confirmed against Sarvam's live docs (docs.sarvam.ai/api-reference-docs/
getting-started/models/bulbul, fetched July 2026): Bulbul's current
11-language list is Hindi, Bengali, Tamil, Telugu, Gujarati, Kannada,
Malayalam, Marathi, Punjabi, Odia, and English. Assamese and Urdu are absent
from that list (the docs don't break out v2 vs v3 separately, so this applies
to both) — this is not a hand-wave, it's confirmed unsupported as of this
check. Route those two languages to the fallback TTS provider transparently;
re-verify against Sarvam's docs periodically in case they add coverage.
"""

from __future__ import annotations

# Confirmed absent from Bulbul's language list — see module docstring.
NOT_SUPPORTED_BY_SARVAM_TTS = {"as", "ur"}


def select_tts_provider(language: str) -> str:
    """Returns "sarvam" or "fallback" — the caller picks the matching client."""
    return "fallback" if language in NOT_SUPPORTED_BY_SARVAM_TTS else "sarvam"
