"""Language detection: 13 pure languages, 5 code-mixed examples, STT-hint
independence, low-confidence routing, and translation policy."""

import pytest

from agent_core.agents.language_agent import (
    LOW_CONFIDENCE_THRESHOLD,
    detect_language,
    disallowed_script_in,
    is_low_confidence,
    resolve_tts_voice_language,
    target_language_mismatch,
)
from agent_core.agents.translation_policy import decide_translation
from agent_core.llm_adapter import LLMRouter

from ._fakes import ScriptedProvider
from ._language_fixtures import CODE_MIXED_CASES, PURE_LANGUAGE_CASES


@pytest.mark.parametrize("text,expected_lang", PURE_LANGUAGE_CASES)
async def test_pure_language_detected_with_no_router_needed(text, expected_lang):
    result = await detect_language(text)

    assert result.language == expected_lang
    assert result.confidence >= LOW_CONFIDENCE_THRESHOLD
    assert result.is_code_mixed is False


@pytest.mark.parametrize("text,expected_lang,needs_llm", CODE_MIXED_CASES)
async def test_code_mixed_input_detected_not_corrected_to_single_language(text, expected_lang, needs_llm):
    router = None
    if needs_llm:
        reply = f'{{"language": "{expected_lang}", "confidence": 0.8, "is_code_mixed": true}}'
        router = LLMRouter([ScriptedProvider([reply])])

    result = await detect_language(text, router=router)

    assert result.language == expected_lang
    assert result.is_code_mixed is True
    assert result.confidence >= LOW_CONFIDENCE_THRESHOLD


async def test_wrong_stt_hint_is_not_trusted():
    tamil_text = "வணக்கம், எப்படி இருக்கிறீர்கள்?"

    result = await detect_language(tamil_text, stt_hint="hi")

    assert result.language == "ta"  # script evidence wins over the wrong hint


async def test_ambiguous_input_routes_to_low_confidence_not_a_guess():
    result = await detect_language("hmm")

    assert result.confidence < LOW_CONFIDENCE_THRESHOLD


async def test_gibberish_escalates_to_llm_when_router_available_but_still_low_if_llm_unsure():
    router = LLMRouter([ScriptedProvider(['{"language": "en", "confidence": 0.3, "is_code_mixed": false}'])])

    result = await detect_language("asdkfj qwop", router=router)

    assert result.confidence < LOW_CONFIDENCE_THRESHOLD


async def test_low_confidence_deterministic_result_escalates_to_llm():
    """Real bug hit live: plain Devanagari Hindi with no explicit marker word
    ("है", "क्या", etc.) scores only 0.36 from the deterministic
    Hindi/Marathi disambiguator — below the clarify threshold — for an
    unambiguous, real Hindi sentence. Previously the `or` chain treated any
    non-None deterministic result as final and never even tried the LLM.
    Escalating on low confidence (not just on total absence) lets the LLM
    resolve it correctly."""
    text = "मुझे पाइथन में वेरिएबल्स के बारे में समझाओ"
    router = LLMRouter([ScriptedProvider(['{"language": "hi", "confidence": 0.9, "is_code_mixed": false}'])])

    result = await detect_language(text, router=router)

    assert result.language == "hi"
    assert result.confidence >= LOW_CONFIDENCE_THRESHOLD


async def test_low_confidence_deterministic_result_kept_if_llm_not_better():
    """The escalation must not blindly prefer the LLM — only take it if it's
    actually more confident than the deterministic guess."""
    text = "मुझे पाइथन में वेरिएबल्स के बारे में समझाओ"
    router = LLMRouter([ScriptedProvider(['{"language": "hi", "confidence": 0.2, "is_code_mixed": false}'])])

    result = await detect_language(text, router=router)

    assert result.confidence == pytest.approx(0.36, abs=0.01)  # kept the deterministic result


def test_translation_not_applied_by_default():
    assert decide_translation("hi") is False


@pytest.mark.parametrize(
    "raw_reply",
    [
        '\n{"language": "te", "confidence": 0.95, "is_code_mixed": true}',
        '\n```json\n{"language": "te", "confidence": 0.95, "is_code_mixed": false}\n```',
        # A real model, asked the same question 3 times, returned this shape
        # once: valid JSON followed by a second, truncated, repeated attempt.
        '\n{"language": "te", "confidence": 0.95, "is_code_mixed": true}\n```json\n{\n  "language": "te",',
    ],
)
async def test_llm_classify_tolerates_real_model_response_shapes(raw_reply):
    """Real bug hit live: asked for "strict JSON only, no prose" three times
    in a row, a real model returned three different shapes — clean JSON,
    JSON wrapped in a markdown fence, and valid JSON followed by rambling
    repeated garbage. A bare `json.loads(raw.strip())` failed on 2 of 3,
    silently dropping detection to "unknown" at 0.2 confidence and
    triggering a bogus clarifying question for an answerable message."""
    router = LLMRouter([ScriptedProvider([raw_reply])])

    result = await detect_language("naku python lo variables gurinchi explain cheyi", router=router)

    assert result.language == "te"
    assert result.confidence == 0.95
    assert result.is_code_mixed in (True, False)
    assert decide_translation("te") is False


def test_translation_applied_when_tool_requires_english():
    assert decide_translation("hi", tool_requires_english=True) is True


def test_native_script_message_uses_its_own_detected_voice():
    """Confirmed live (round-trip WER testing): Sarvam's Indic voices
    handle native-script text cleanly -- WER 0.00 across all 11 testable
    languages. A native-script message should use its own voice."""
    assert resolve_tts_voice_language("మీరు ఎలా ఉన్నారు?", "te") == "te"


def test_romanized_message_falls_back_to_english_voice():
    """Real bug hit live: Sarvam's Indic voices badly mangle romanized
    (Latin-script) text even when the target language is genuinely correct
    -- WER 0.75-1.25 on real romanized Telugu/Hindi/English mixes, "Bro"
    mispronounced as "Uru" in two independent test runs. The text-language
    detection (still Telugu here) is untouched -- only the voice falls
    back."""
    assert resolve_tts_voice_language("Bro meeting ki vasthunnava?", "te") == "en"


def test_english_target_is_never_second_guessed():
    assert resolve_tts_voice_language("Hello, how are you?", "en") == "en"


def test_unknown_detected_language_resolves_to_english_voice():
    assert resolve_tts_voice_language("anything at all", "unknown") == "en"
    assert resolve_tts_voice_language("anything at all", None) == "en"


def test_mixed_script_message_uses_the_majority_script():
    """More Telugu characters than Latin -- should still use the native
    voice; the threshold is majority, not purity (an occasional embedded
    English word, e.g. "office", shouldn't force an English voice)."""
    assert resolve_tts_voice_language("నేను ఈరోజు office కి వెళ్తాను ఎందుకంటే మీటింగ్ ఉంది", "te") == "te"


def test_is_low_confidence_matches_the_threshold():
    """Architecture gap closed: this exact check used to be duplicated,
    once in supervisor/graph.py's route_after_language and once inline in
    api/main.py's /chat/stream -- both now defer to this one function."""
    assert is_low_confidence(LOW_CONFIDENCE_THRESHOLD - 0.01) is True
    assert is_low_confidence(LOW_CONFIDENCE_THRESHOLD) is False
    assert is_low_confidence(None) is True  # no confidence at all -- treat as low, not high


def test_disallowed_script_detects_chinese():
    """Regression test for a real live incident: self-check logged
    "VIOLATION: response is in Chinese, not the target language English" --
    none of the 13 supported languages ever legitimately use CJK script."""
    assert disallowed_script_in("你好，这是中文回复。") == "Chinese"


def test_disallowed_script_ignores_every_supported_language_and_code_mixing():
    assert disallowed_script_in("Hello, how are you?") is None
    assert disallowed_script_in("నేను ఈరోజు office కి వెళ్తాను") is None
    assert disallowed_script_in("यह हिंदी में एक वाक्य है") is None


def test_target_language_mismatch_catches_telugu_reply_for_english_target():
    """Regression test for a real live incident: an English-target turn
    answered entirely in Telugu (turn_trace showed the reply for "what is
    inference" came back as native Telugu text with zero English)."""
    telugu_reply = (
        "Inference అంటే ఒక నిర్ధారణ లేదా ఒక ముగింపు, ఇది మనకు తెలిసినవిషయాల ఆధారంగా చేసే అనుమితి."
    )
    assert target_language_mismatch(telugu_reply, "en", is_code_mixed=False) is True


def test_target_language_mismatch_catches_telugu_english_codemix_for_english_target():
    """Regression test for the other live incident: turn_trace logged
    "VIOLATION: draft is not in the target language English, it is in
    Telugu-English code-mixed" -- English never legitimately code-mixes
    with an Indic script (only the reverse direction is a real style)."""
    reply = (
        "Machine learning models డేటా నుండి నేర్చుకుంటాయి మరియు కాలక్రమేణా వాటి ఖచ్చితత్వాన్ని "
        "మెరుగుపరుస్తాయి, ఇది చాలా ఉపయోగకరంగా ఉంటుంది."
    )
    assert target_language_mismatch(reply, "en", is_code_mixed=False) is True


def test_target_language_mismatch_allows_a_genuine_english_reply():
    assert target_language_mismatch("Hello, how can I help you today?", "en", is_code_mixed=False) is False


def test_target_language_mismatch_allows_legitimate_indic_english_codemixing():
    """The one direction that IS a real, documented style: an Indic target
    naturally code-mixed with English words."""
    reply = "నేను ఈరోజు office కి వెళ్తాను ఎందుకంటే మీటింగ్ ఉంది"
    assert target_language_mismatch(reply, "te", is_code_mixed=True) is False


def test_target_language_mismatch_catches_wrong_indic_script_for_indic_target():
    hindi_reply = "यह एक हिंदी वाक्य है जो तेलुगु के बजाय आया"
    assert target_language_mismatch(hindi_reply, "te", is_code_mixed=False) is True


def test_target_language_mismatch_ignores_a_short_ambiguous_reply():
    """Too little script signal to safely judge -- avoids a false positive
    on a short reply like "42" or "Yes." with no real language content."""
    assert target_language_mismatch("42", "te", is_code_mixed=False) is False
    assert target_language_mismatch("OK", "hi", is_code_mixed=False) is False


def test_target_language_mismatch_tolerates_an_occasional_loanword():
    """An occasional embedded proper noun/loanword must not trip this --
    only a clear script-majority miss does."""
    reply = "మీ ప్రశ్నకు సమాధానం ఇక్కడ ఉంది. MTouch Labs వారి సేవలు చాలా బాగున్నాయి."
    assert target_language_mismatch(reply, "te", is_code_mixed=True) is False
