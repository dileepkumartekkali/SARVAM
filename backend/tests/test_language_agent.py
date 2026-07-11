"""Language detection: 13 pure languages, 5 code-mixed examples, STT-hint
independence, low-confidence routing, and translation policy."""

import pytest

from agent_core.agents.language_agent import LOW_CONFIDENCE_THRESHOLD, detect_language
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
