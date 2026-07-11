"""Response chunker for TTS (S2S plan §4): splits streaming LLM text into
TTS-safe chunks as sentence boundaries appear — not after the full response
is generated, so time-to-first-audio stays low.

Rules, in priority order:
1. Split on sentence boundaries, language-aware — not naive `.` splitting.
   Recognizes ASCII `. ! ?`, Devanagari danda `।`/`॥`, and Urdu `؟`, so Indic
   punctuation and code-mixed sentences aren't mis-split.
2. Fall back to clause-level (comma/conjunction) only if a "sentence" exceeds
   ~25 words — most sentences never need this.
3. Never split mid-word or mid-number. "₹4,500" must stay one token: the
   splitter treats a currency/number run (digits, embedded commas/decimal
   points, currency symbol) as atomic, so an embedded comma is never mistaken
   for a clause boundary.
4. The very first chunk is capped small (5-10 words) to minimize
   time-to-first-audio; later chunks use the normal sentence/clause rules.
"""

from __future__ import annotations

import re
from typing import AsyncIterator

_SENTENCE_END = re.compile(r"[.!?।॥؟]")
# Splits on a comma (dropped — commas aren't spoken) or on the whitespace
# immediately before a conjunction (a lookahead, so the conjunction itself
# stays attached to the next clause instead of being discarded).
_CLAUSE_BREAK = re.compile(r",\s*|\s+(?=(?:and|but|or|so|because|then)\b)", re.IGNORECASE)

# A currency/number run: optional currency symbol, digits with embedded
# thousands separators (comma) or a decimal point, e.g. "₹4,500" or "3.14".
# Matched first and masked before clause-splitting so its internal commas are
# never treated as split points.
_NUMBER_RUN = re.compile(r"[₹$€£]?\d[\d,]*(?:\.\d+)?")

MAX_WORDS_BEFORE_CLAUSE_FALLBACK = 25
FIRST_CHUNK_TARGET_WORDS = (5, 10)


def _mask_number_runs(text: str) -> tuple[str, list[str]]:
    """Replaces each number/currency run with a placeholder token so clause
    splitting never lands inside one; returns the masked text and the
    original runs to restore afterward."""
    runs: list[str] = []

    def _replace(m: re.Match) -> str:
        runs.append(m.group(0))
        return f"\x00{len(runs) - 1}\x00"

    return _NUMBER_RUN.sub(_replace, text), runs


def _unmask_number_runs(text: str, runs: list[str]) -> str:
    for i, run in enumerate(runs):
        text = text.replace(f"\x00{i}\x00", run)
    return text


def _split_into_clauses(sentence: str) -> list[str]:
    """Splits an over-long sentence on commas/conjunctions, protecting number
    runs from being split on their internal commas."""
    masked, runs = _mask_number_runs(sentence)
    pieces = _CLAUSE_BREAK.split(masked)
    return [_unmask_number_runs(p, runs).strip() for p in pieces if p.strip()]


def _finalize_sentence(sentence: str) -> list[str]:
    word_count = len(sentence.split())
    if word_count <= MAX_WORDS_BEFORE_CLAUSE_FALLBACK:
        return [sentence.strip()]
    return _split_into_clauses(sentence)


async def chunk_stream(text_deltas: AsyncIterator[str]) -> AsyncIterator[str]:
    """Consumes incremental LLM text deltas and yields TTS-ready chunks as
    sentence boundaries appear in the accumulated buffer — never buffers the
    whole response before yielding anything.
    """
    buffer = ""
    first_chunk_emitted = False

    async for delta in text_deltas:
        buffer += delta

        while True:
            masked, runs = _mask_number_runs(buffer)
            match = _SENTENCE_END.search(masked)
            if match is None:
                break
            split_at = match.end()
            sentence = _unmask_number_runs(masked[:split_at], runs)
            buffer = _unmask_number_runs(masked[split_at:], runs)

            if not first_chunk_emitted:
                first_chunk_emitted = True
                words = sentence.split()
                if len(words) > FIRST_CHUNK_TARGET_WORDS[1]:
                    # First sentence is long — emit only a short lead-in chunk
                    # for fast TTFB, keep the remainder for normal processing.
                    head = " ".join(words[: FIRST_CHUNK_TARGET_WORDS[1]])
                    tail = " ".join(words[FIRST_CHUNK_TARGET_WORDS[1] :])
                    yield head.strip()
                    for piece in _finalize_sentence(tail):
                        yield piece
                    continue

            for piece in _finalize_sentence(sentence):
                yield piece

    remainder = buffer.strip()
    if remainder:
        for piece in _finalize_sentence(remainder):
            yield piece
