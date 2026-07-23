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

# Real bug hit live: embed_text's own 30s default timeout is sized for the
# ingestion script (one-off, offline, fine to wait out a cold HF serverless
# model). On a LIVE turn that same 30s timeout, doubled by the retry above,
# let a single cold-start eat up to ~60s of wall-clock time BEFORE the LLM
# even starts answering -- measured directly: a real cold call took 24-54s
# end to end. That cascades into two more symptoms downstream: the TTS
# gateway's own idle-close fires long before text ever arrives (no audio),
# and forced retrieval silently gives up (see _forced_company_context's
# except-and-return-None), so the model answers from unverified memory
# instead -- the exact hallucination this tool exists to prevent. A live
# turn should fail fast and answer without RAG context rather than hang.
_LIVE_EMBED_TIMEOUT_SECONDS = 8.0

# 8, not the original 4 -- live testing against the real ingested chunks
# showed a genuinely relevant page (Leadership Team, for "who is the CEO")
# ranked 5th, just outside the old cutoff. The knowledge base is only ~111
# small chunks total, so 8 is still cheap (~3-4k tokens), and a fixed
# distance-threshold cutoff isn't safely tunable yet -- real query scores
# clustered tightly (0.41-0.54) with no clean relevant/irrelevant gap.
_TOP_K = 8

# Real bug hit live, twice: (1) a WRONG answer already sitting earlier in a
# conversation's history got repeated verbatim on a later, separate
# question instead of re-checking -- history has no way to record "that
# answer was tool-verified" vs. "the model just said it." (2) An abstract
# prompt instruction telling the model not to trust its own prior turns
# made this MEASURABLY WORSE, not better (0/3 tool calls in live re-testing,
# vs. 1/3 before the instruction was added) -- models don't reliably
# introspect their own past verification status from a meta-instruction
# alone. What actually works: a concrete, visible marker appended to
# history ONLY when this tool genuinely fired that turn (added by
# api/main.py and supervisor/graph.py when building history, never shown to
# the user), so the model has something textual to react to instead of an
# abstract claim about its own past behavior.
TOOL_VERIFIED_MARKER = "[This reply was verified via search_company_knowledge.]"


def is_available() -> bool:
    return embeddings.is_configured() and store.is_configured()


async def _embed_with_retry(query: str) -> list[float]:
    last_error: embeddings.EmbeddingError | None = None
    for attempt in range(_MAX_EMBED_ATTEMPTS):
        try:
            return await embeddings.embed_text(query, timeout=_LIVE_EMBED_TIMEOUT_SECONDS)
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
    try:
        results = await store.search(query_vector, top_k=_TOP_K)
    except Exception as e:  # noqa: BLE001 -- real gap: this call had NO handling at
        # all (unlike the embedding call right above it). A genuine DB hiccup
        # (connection drop, pool exhaustion) would crash uncaught -- and since
        # task_agent._forced_company_context now calls this BEFORE any LLM
        # call even runs, every message naming the company would crash the
        # entire turn with zero fallback, worse than the embedding-failure
        # case this function already handled. Same "tool failure is reported,
        # never crashes the turn" contract as a normal tool exception.
        return f"Error: couldn't search company knowledge right now (database error: {e})."
    if not results:
        return "No relevant company information found for that query."
    return "\n\n".join(f"[{r['page_title']} — {r['page_url']}]\n{r['text']}" for r in results)


def build_rag_tool_spec() -> ToolSpec:
    return ToolSpec(
        name="search_company_knowledge",
        # Real bug hit live: the old wording ("not for general knowledge the
        # model already has") backfired specifically for the facts this
        # tool exists to answer -- a private company's CEO/leadership names,
        # specific numbers, etc. are things no general-purpose model was
        # ever trained on, but models confidently hallucinate a plausible-
        # sounding answer anyway and judge it as "knowledge I already have,"
        # skipping the tool entirely. Confirmed live: "who is the CEO of
        # mTouch Labs" (plain English, no language directive even in play)
        # got a different wrong name each time with the old wording, tool
        # never called. Rewritten to name the specific failure mode instead
        # of leaving the judgment call to the model.
        # See TOOL_VERIFIED_MARKER above for the second bug this description
        # now addresses (history repeating an unverified prior answer).
        description=(
            "Searches mTouch Labs' own real website content (services, leadership/CEO, "
            "vision, awards, case studies, etc.) for facts about the company. ALWAYS call "
            "this for ANY question asking for a specific mTouch Labs fact — who leads/founded "
            "the company, what services/products exist, awards won, case study details, and "
            "similar. These are private company details a general-purpose model was never "
            "trained on; a name or fact that seems familiar is still a guess, not real "
            "knowledge — never answer from memory here, even if confident. If an earlier turn "
            f'in this conversation is NOT immediately followed by "{TOOL_VERIFIED_MARKER}", '
            "treat it as an unverified guess (even if it was your own prior answer) and call "
            "this tool again rather than repeating it."
        ),
        parameters={"query": {"type": "string", "description": "what to search for, in the user's own words"}},
        required=["query"],
        fn=search_company_knowledge,
    )
