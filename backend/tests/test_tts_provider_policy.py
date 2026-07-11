"""Confirmed against Sarvam's live docs (July 2026): Bulbul's 11-language list
excludes Assamese and Urdu. This test locks in that routing decision so a
regression (e.g. someone "fixing" the set without re-checking the docs)
doesn't silently break the fallback path."""

from agent_core.speech.tts_provider_policy import select_tts_provider


def test_assamese_routes_to_fallback():
    assert select_tts_provider("as") == "fallback"


def test_urdu_routes_to_fallback():
    assert select_tts_provider("ur") == "fallback"


def test_supported_languages_route_to_sarvam():
    for lang in ("hi", "te", "ta", "kn", "ml", "mr", "gu", "pa", "bn", "or", "en"):
        assert select_tts_provider(lang) == "sarvam"
