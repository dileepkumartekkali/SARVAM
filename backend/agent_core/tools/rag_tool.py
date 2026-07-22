"""The one RAG-facing tool: lets the LLM pull real mTouch Labs company facts
(services, leadership, awards, etc. -- see agent_core/rag/) into its answer.

Registered as a normal tool (not an always-on pre-retrieval step) so a plain
"hi" or "thanks" never pays an embedding-call+DB-query cost it didn't need --
the LLM only reaches for it when a question actually looks like it needs
company-specific knowledge, same reasoning that kept language detection
tiered instead of merged into every turn.
"""

from __future__ import annotations

import asyncio

from ..rag import embeddings, store
from .registry import ToolSpec

# One retry, not zero -- real gap: embeddings.EmbeddingError already carried
# a `retriable` flag (true for HF's transient 503 "model loading" / 429
# rate-limit responses) but nothing anywhere ever actually retried on it, so
# a purely transient hiccup permanently failed the whole tool call mid-turn,
# which is exactly the shape of the "trouble getting an answer" apology
# users were seeing. A short fixed delay, not chat_store.record_turn's 3
# attempts -- this runs inline during a live turn, where the whole point of
# streaming is not adding wasted latency.
_MAX_EMBED_ATTEMPTS = 2
_RETRY_DELAY_SECONDS = 0.5

# 8, not the original 4 -- live testing against the real ingested chunks
# showed a genuinely relevant page (Leadership Team, for "who is the CEO")
# ranked 5th, just outside the old cutoff. The knowledge base is only ~111
# small chunks total, so 8 is still cheap (~3-4k tokens), and a fixed
# distance-threshold cutoff isn't safely tunable yet -- real query scores
# clustered tightly (0.41-0.54) with no clean relevant/irrelevant gap.
_TOP_K = 8


def is_available() -> bool:
    return embeddings.is_configured() and store.is_configured()


async def _embed_with_retry(query: str) -> list[float]:
    last_error: embeddings.EmbeddingError | None = None
    for attempt in range(_MAX_EMBED_ATTEMPTS):
        try:
            return await embeddings.embed_text(query)
        except embeddings.EmbeddingError as e:
            last_error = e
            if not e.retriable or attempt == _MAX_EMBED_ATTEMPTS - 1:
                raise
            await asyncio.sleep(_RETRY_DELAY_SECONDS)
    raise last_error  # unreachable, satisfies static analysis


async def search_company_knowledge(query: str) -> str:
    """Embeds `query` (whatever language it's in -- bge-m3 is cross-lingual,
    no translation step needed) and returns the most relevant chunks of
    mTouch Labs' own website content for the LLM to answer from."""
    try:
        query_vector = await _embed_with_retry(query)
    except embeddings.EmbeddingError as e:
        return f"Error: couldn't search company knowledge right now ({e})."
    results = await store.search(query_vector, top_k=_TOP_K)
    if not results:
        return "No relevant company information found for that query."
    return "\n\n".join(f"[{r['page_title']} — {r['page_url']}]\n{r['text']}" for r in results)


def build_rag_tool_spec() -> ToolSpec:
    return ToolSpec(
        name="search_company_knowledge",
        description=(
            "Searches mTouch Labs' own website content (services, leadership, "
            "vision, awards, etc.) for facts relevant to the user's question. "
            "Use this for anything specific to mTouch Labs as a company -- "
            "not for general knowledge the model already has."
        ),
        parameters={"query": {"type": "string", "description": "what to search for, in the user's own words"}},
        required=["query"],
        fn=search_company_knowledge,
    )
