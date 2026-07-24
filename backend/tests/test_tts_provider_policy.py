"""Confirmed against Sarvam's live docs (July 2026): Bulbul's 11-language list
excludes Assamese and Urdu. This test locks in that routing decision so a
regression (e.g. someone "fixing" the set without re-checking the docs)
doesn't silently break the fallback path."""

from agent_core.speech.tts_provider_policy import languages_missing_tts_coverage, select_tts_provider


def test_assamese_routes_to_fallback():
    assert select_tts_provider("as") == "fallback"


def test_urdu_routes_to_fallback():
    assert select_tts_provider("ur") == "fallback"


def test_supported_languages_route_to_sarvam():
    for lang in ("hi", "te", "ta", "kn", "ml", "mr", "gu", "pa", "bn", "or", "en"):
        assert select_tts_provider(lang) == "sarvam"


def test_missing_tts_coverage_flags_assamese_and_urdu_when_fallback_unconfigured():
    """Architecture gap closed: previously nothing checked at boot whether
    every supported language actually has a working TTS path -- Assamese/
    Urdu route to the Azure fallback, and if that's never configured, those
    two languages silently have no voice output at all."""
    missing = languages_missing_tts_coverage({"en", "te", "as", "ur"}, fallback_configured=False)
    assert missing == ["as", "ur"]


def test_missing_tts_coverage_is_empty_when_fallback_is_configured():
    missing = languages_missing_tts_coverage({"en", "te", "as", "ur"}, fallback_configured=True)
    assert missing == []


def test_missing_tts_coverage_ignores_languages_that_dont_need_the_fallback():
    missing = languages_missing_tts_coverage({"en", "te", "hi"}, fallback_configured=False)
    assert missing == []
