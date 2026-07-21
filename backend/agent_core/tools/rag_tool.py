"""The one RAG-facing tool: lets the LLM pull real mTouch Labs company facts
(services, leadership, awards, etc. -- see agent_core/rag/) into its answer.

Registered as a normal tool (not an always-on pre-retrieval step) so a plain
"hi" or "thanks" never pays an embedding-call+DB-query cost it didn't need --
the LLM only reaches for it when a question actually looks like it needs
company-specific knowledge, same reasoning that kept language detection
tiered instead of merged into every turn.
"""

from __future__ import annotations

from ..rag import embeddings, store
from .registry import ToolSpec

_TOP_K = 4


def is_available() -> bool:
    return embeddings.is_configured() and store.is_configured()


async def search_company_knowledge(query: str) -> str:
    """Embeds `query` (whatever language it's in -- bge-m3 is cross-lingual,
    no translation step needed) and returns the most relevant chunks of
    mTouch Labs' own website content for the LLM to answer from."""
    try:
        query_vector = await embeddings.embed_text(query)
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
