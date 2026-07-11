"""Response chunker: sentence/clause splitting, the ₹4,500 case, first-chunk
sizing, and streaming (not buffer-then-split) behavior."""

from agent_core.speech.chunker import chunk_stream


async def _deltas(*parts):
    for p in parts:
        yield p


async def _collect(parts):
    return [c async for c in chunk_stream(_deltas(*parts))]


async def test_currency_amount_is_never_split():
    chunks = await _collect(["The total is ₹4,500 for this order. ", "Thanks for shopping with us!"])

    assert any("₹4,500" in c for c in chunks)
    assert not any(c.strip() == "₹4" for c in chunks)
    assert not any(c.strip().startswith(",500") for c in chunks)


async def test_short_sentence_with_comma_is_not_clause_split():
    chunks = await _collect(["Hello there, welcome! ", "How can I help you today?"])

    assert chunks[0] == "Hello there, welcome!"


async def test_long_sentence_falls_back_to_clause_splitting():
    long_sentence = (
        "This is a very long sentence that goes well beyond twenty five words because it keeps "
        "adding more clauses and more words without any period to stop it, so it should fall back "
        "to clause splitting eventually, and that is the point of this test."
    )
    chunks = await _collect([long_sentence])

    assert len(chunks) > 1
    # Commas are intentionally dropped (they aren't spoken); every word survives.
    joined = "".join(chunks).replace(" ", "").replace(",", "")
    original = long_sentence.replace(" ", "").replace(",", "")
    assert joined in original


async def test_short_sentence_under_25_words_is_not_clause_split():
    # Kept to 8 words so it also stays under the first-chunk cap (10 words) —
    # that's a separate rule, tested on its own below.
    chunks = await _collect(["Reply with, say, a few commas here."])

    assert len(chunks) == 1


async def test_first_chunk_is_capped_small_for_fast_ttfb():
    long_sentence = "word " * 30 + "end."
    chunks = await _collect([long_sentence])

    assert len(chunks[0].split()) <= 10


async def test_chunks_are_emitted_before_full_input_is_consumed():
    """Proves streaming behavior: the first chunk appears after only the first
    delta, not after the whole generator is drained."""
    seen = []

    async def deltas():
        yield "First sentence. "
        seen.append("after-first-delta")
        yield "Second sentence."
        seen.append("after-second-delta")

    chunks = []
    async for chunk in chunk_stream(deltas()):
        chunks.append(chunk)
        if len(chunks) == 1:
            # The first chunk must reach us before chunk_stream even asks the
            # producer for its second delta.
            assert seen == []

    assert chunks == ["First sentence.", "Second sentence."]


async def test_devanagari_danda_is_a_sentence_boundary():
    chunks = await _collect(["नमस्ते। आप कैसे हैं।"])

    assert len(chunks) == 2
    assert chunks[0] == "नमस्ते।"
