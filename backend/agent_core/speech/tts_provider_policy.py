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


def languages_missing_tts_coverage(supported_languages: set[str], *, fallback_configured: bool) -> list[str]:
    """Architecture gap closed: nothing previously checked, at boot, whether
    every language this system claims to support ("SUPPORTED_LANGUAGES" in
    language_agent.py) actually has a WORKING TTS path. Assamese/Urdu route
    to the Azure fallback provider by design (see module docstring) — but if
    that fallback was never actually configured (no AZURE_SPEECH_KEY/REGION),
    those two languages silently have no voice output at all, discoverable
    previously only by a real user hitting it live. Returns the supported
    languages that would fail today, so the caller can log it loudly at
    startup instead of leaving it to be discovered in production."""
    if fallback_configured:
        return []
    return sorted(lang for lang in supported_languages if select_tts_provider(lang) == "fallback")
