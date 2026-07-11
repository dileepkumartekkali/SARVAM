"""Language detection, confidence scoring, and code-mix flagging.

Detection is deliberately layered so the common cases never touch the network
(per speech_to_speech_plan.md — "don't trust STT's language guess blindly," and
per general latency/cost sense): a pure single-script sentence in any of the 13
supported languages is identified from Unicode script alone, with zero
ambiguity for 9 of them and a tiny lexical-marker check for the two script
pairs that genuinely can't be told apart by codepoint (Hindi/Marathi share
Devanagari; Bengali/Assamese share Bengali script — Assamese is distinguished
by two characters, ৰ/ৱ, that Bengali doesn't use).

Fully-romanized code-mixed input (e.g. "Bro meeting ki vasthunnava?") has no
script signal at all — every character is plain Latin. For that case we try a
tiny hand-curated keyword list first (cheap, no network), and only fall back
to an LLM classification call when nothing deterministic matched anything.

The STT hint (`stt_hint`) is never used to decide the result — only passed to
the LLM classifier as a labeled, explicitly-unverified signal. If nothing
matches and no router is available, detection returns "unknown" at very low
confidence rather than adopting the hint — the graph routes that to a
clarifying question instead of guessing.

ponytail: the romanized-keyword list only covers Telugu markers today (the one
confirmed example). Extending it per-language is a losing battle to maintain —
every other romanized case relies on the LLM fallback below, which is where
real coverage should grow.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass

from ..llm_adapter.base import LLMProviderError, LLMRouter

SUPPORTED_LANGUAGES = {"en", "hi", "te", "ta", "kn", "ml", "mr", "gu", "pa", "bn", "or", "as", "ur"}

# Natural-language names for the per-turn language directive (task_agent's
# `_language_directive`) — a code alone ("te") reads oddly in an instruction;
# a name ("Telugu") is unambiguous to the model.
LANGUAGE_NAMES: dict[str, str] = {
    "en": "English",
    "hi": "Hindi",
    "te": "Telugu",
    "ta": "Tamil",
    "kn": "Kannada",
    "ml": "Malayalam",
    "mr": "Marathi",
    "gu": "Gujarati",
    "pa": "Punjabi",
    "bn": "Bengali",
    "or": "Odia",
    "as": "Assamese",
    "ur": "Urdu",
}

# Below this, the caller (the LangGraph supervisor) should ask a clarifying
# question rather than answer in a possibly-wrong language.
LOW_CONFIDENCE_THRESHOLD = 0.5


@dataclass
class LanguageDetectionResult:
    language: str
    confidence: float
    is_code_mixed: bool


# --- Unicode-script detection -----------------------------------------------

_SCRIPT_RANGES: list[tuple[int, int, str]] = [
    (0x0600, 0x06FF, "arabic"),
    (0x0900, 0x097F, "devanagari"),
    (0x0980, 0x09FF, "bengali"),
    (0x0A00, 0x0A7F, "gurmukhi"),
    (0x0A80, 0x0AFF, "gujarati"),
    (0x0B00, 0x0B7F, "oriya"),
    (0x0B80, 0x0BFF, "tamil"),
    (0x0C00, 0x0C7F, "telugu"),
    (0x0C80, 0x0CFF, "kannada"),
    (0x0D00, 0x0D7F, "malayalam"),
]

_SCRIPT_TO_LANG = {
    "arabic": "ur",
    "gurmukhi": "pa",
    "gujarati": "gu",
    "oriya": "or",
    "tamil": "ta",
    "telugu": "te",
    "kannada": "kn",
    "malayalam": "ml",
}

_HI_MARKERS = {"है", "हैं", "आप", "कैसे", "नहीं", "क्या"}
_MR_MARKERS = {"आहे", "आहात", "तुम्ही", "काय", "नाही", "मी"}
_ASSAMESE_ONLY_CHARS = ("ৰ", "ৱ")  # ৰ, ৱ — absent from standard Bengali


def _script_of_char(ch: str) -> str | None:
    cp = ord(ch)
    for lo, hi, name in _SCRIPT_RANGES:
        if lo <= cp <= hi:
            return name
    if ch.isascii() and ch.isalpha():
        return "latin"
    return None


def _script_profile(text: str) -> Counter:
    profile: Counter = Counter()
    for ch in text:
        script = _script_of_char(ch)
        if script:
            profile[script] += 1
    return profile


def _disambiguate_devanagari(text: str) -> tuple[str, float]:
    hi_hits = sum(1 for w in _HI_MARKERS if w in text)
    mr_hits = sum(1 for w in _MR_MARKERS if w in text)
    if hi_hits == 0 and mr_hits == 0:
        return "hi", 0.4  # no markers either way — weak default guess
    if hi_hits >= mr_hits:
        return "hi", 1.0 if mr_hits == 0 else 0.75
    return "mr", 1.0 if hi_hits == 0 else 0.75


def _disambiguate_bengali(text: str) -> tuple[str, float]:
    if any(ch in text for ch in _ASSAMESE_ONLY_CHARS):
        return "as", 1.0
    return "bn", 0.85


_AMBIGUOUS_SCRIPTS = {"devanagari": _disambiguate_devanagari, "bengali": _disambiguate_bengali}


def _script_based_detect(text: str) -> LanguageDetectionResult | None:
    profile = _script_profile(text)
    total = sum(profile.values())
    if total == 0:
        return None

    indic = {k: v for k, v in profile.items() if k != "latin"}
    if not indic:
        return None  # purely Latin — script alone can't identify the language

    dominant_script, _ = max(indic.items(), key=lambda kv: kv[1])
    latin_count = profile.get("latin", 0)
    is_code_mixed = latin_count >= 2 and latin_count / total > 0.15

    if dominant_script in _AMBIGUOUS_SCRIPTS:
        lang, disambig_confidence = _AMBIGUOUS_SCRIPTS[dominant_script](text)
    else:
        lang, disambig_confidence = _SCRIPT_TO_LANG[dominant_script], 1.0

    return LanguageDetectionResult(
        language=lang, confidence=round(0.9 * disambig_confidence, 2), is_code_mixed=is_code_mixed
    )


# --- Romanized code-mix keywords (deterministic, no network) ----------------

_ROMANIZED_MARKERS: dict[str, set[str]] = {
    "te": {"vasthunnava", "vachindi", "enti", "chesav", "cheppu", "unnav"},
}


def _romanized_keyword_detect(text: str) -> LanguageDetectionResult | None:
    lowered = text.lower()
    for lang, markers in _ROMANIZED_MARKERS.items():
        hits = sum(1 for m in markers if m in lowered)
        if hits:
            return LanguageDetectionResult(
                language=lang, confidence=min(0.85, 0.5 + 0.15 * hits), is_code_mixed=True
            )
    return None


# --- Plain English (Latin script, no other signal) --------------------------

_EN_STOPWORDS = {"the", "is", "are", "you", "how", "what", "please", "thanks", "hello", "hi", "hey", "yes", "no"}


def _plain_english_detect(text: str) -> LanguageDetectionResult | None:
    words = re.findall(r"[a-zA-Z']+", text.lower())
    hits = sum(1 for w in words if w in _EN_STOPWORDS)
    if hits == 0:
        return None
    return LanguageDetectionResult(language="en", confidence=min(0.9, round(0.5 + 0.15 * hits, 2)), is_code_mixed=False)


# --- LLM fallback (last resort, only when nothing deterministic matched) ---

_LANGUAGE_CLASSIFY_SYSTEM = (
    "You are a precise language identification classifier for a voice assistant "
    "supporting these ISO 639-1 codes: en, hi, te, ta, kn, ml, mr, gu, pa, bn, or, "
    "as, ur — including code-mixed input. Reply with strict JSON only, no prose: "
    '{"language": "<code>", "confidence": <0-1 float>, "is_code_mixed": <bool>}.'
)


def _extract_json_object(raw: str) -> str | None:
    """Finds the first complete `{...}` object in `raw` by brace-depth
    counting, ignoring anything before or after it. Verified live this is
    necessary, not defensive-for-its-own-sake: asked three times for the
    exact same "strict JSON only, no prose" instruction, a real model
    returned three different shapes — clean JSON, JSON wrapped in a
    ```json fence, and valid JSON followed by rambling repeated garbage.
    A bare `json.loads(raw.strip())` failed on 2 of the 3, silently
    dropping detection to "unknown" at 0.2 confidence and triggering a
    bogus clarifying question for a perfectly answerable message.
    """
    start = raw.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(raw)):
        if raw[i] == "{":
            depth += 1
        elif raw[i] == "}":
            depth -= 1
            if depth == 0:
                return raw[start : i + 1]
    return None


async def _llm_classify(text: str, router: LLMRouter, *, stt_hint: str | None) -> LanguageDetectionResult | None:
    hint_line = f"\nUnverified STT language hint (do not trust blindly): {stt_hint}" if stt_hint else ""
    try:
        raw = await router.complete_with_fallback(
            [{"role": "user", "content": f"{text}{hint_line}"}],
            system=_LANGUAGE_CLASSIFY_SYSTEM,
            max_tokens=60,
        )
        json_text = _extract_json_object(raw)
        if json_text is None:
            return None
        payload = json.loads(json_text)
        lang = payload["language"]
        if lang not in SUPPORTED_LANGUAGES:
            return None
        return LanguageDetectionResult(
            language=lang,
            confidence=float(payload.get("confidence", 0.7)),
            is_code_mixed=bool(payload.get("is_code_mixed", False)),
        )
    except (LLMProviderError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


async def detect_language(
    text: str, *, stt_hint: str | None = None, router: LLMRouter | None = None
) -> LanguageDetectionResult:
    result = _script_based_detect(text) or _romanized_keyword_detect(text) or _plain_english_detect(text)
    # Real bug hit live: plain Devanagari Hindi with no explicit Hindi/Marathi
    # marker word (_HI_MARKERS/_MR_MARKERS) scores only 0.36 confidence from
    # `_disambiguate_devanagari`'s weak default guess — below
    # LOW_CONFIDENCE_THRESHOLD, triggering a bogus clarifying question for a
    # perfectly unambiguous, real Hindi sentence. The `or` chain above only
    # escalates to the LLM when the deterministic path found NOTHING at
    # all — a low-confidence result still short-circuited it. Escalating
    # whenever confidence is below threshold (not just when result is None)
    # lets the LLM resolve exactly this kind of ambiguity; the deterministic
    # result is kept if the LLM call fails or isn't more confident.
    if router is not None and (result is None or result.confidence < LOW_CONFIDENCE_THRESHOLD):
        llm_result = await _llm_classify(text, router, stt_hint=stt_hint)
        if llm_result is not None and (result is None or llm_result.confidence > result.confidence):
            result = llm_result
    if result is None:
        result = LanguageDetectionResult(language="unknown", confidence=0.2, is_code_mixed=False)
    return result
