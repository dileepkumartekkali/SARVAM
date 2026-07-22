"""agent_core.tools.rag_tool -- the LLM-callable search wired to the RAG
pipeline. Embeddings/store are monkeypatched here (already independently
tested in test_rag_embeddings.py/test_rag_store.py); this file only checks
the tool's own glue: formatting, the no-results case, and the embedding-
failure case.
"""

import pytest

from agent_core.rag import embeddings
from agent_core.tools import rag_tool


async def test_search_formats_results_with_title_and_url(monkeypatch):
    async def fake_embed_text(query, **kwargs):
        return [0.1, 0.2]

    async def fake_search(query_vector, *, top_k):
        return [
            {"chunk_id": "about#0", "page_url": "https://mtouchlabs.com/about", "page_title": "About", "text": "We build software.", "distance": 0.1},
        ]

    monkeypatch.setattr(rag_tool.embeddings, "embed_text", fake_embed_text)
    monkeypatch.setattr(rag_tool.store, "search", fake_search)

    result = await rag_tool.search_company_knowledge("what does the company do")

    assert "About" in result
    assert "https://mtouchlabs.com/about" in result
    assert "We build software." in result


async def test_search_reports_no_results_plainly(monkeypatch):
    async def fake_embed_text(query, **kwargs):
        return [0.1, 0.2]

    async def fake_search(query_vector, *, top_k):
        return []

    monkeypatch.setattr(rag_tool.embeddings, "embed_text", fake_embed_text)
    monkeypatch.setattr(rag_tool.store, "search", fake_search)

    result = await rag_tool.search_company_knowledge("something irrelevant")

    assert "no relevant" in result.lower()


async def test_search_surfaces_a_database_failure_without_raising(monkeypatch):
    """Real gap caught by a direct question about edge cases: store.search
    had NO exception handling at all, unlike the embedding call right above
    it. A genuine DB hiccup (connection drop, pool exhaustion) would crash
    uncaught -- and since this now also runs as FORCED retrieval before any
    LLM call, every message naming the company would crash the whole turn
    with zero fallback."""
    async def fake_embed_text(query, **kwargs):
        return [0.1, 0.2]

    async def fake_search(query_vector, *, top_k):
        raise ConnectionError("connection to Postgres lost")

    monkeypatch.setattr(rag_tool.embeddings, "embed_text", fake_embed_text)
    monkeypatch.setattr(rag_tool.store, "search", fake_search)

    result = await rag_tool.search_company_knowledge("who is the CEO")

    assert "error" in result.lower()


async def test_search_surfaces_embedding_failure_without_raising(monkeypatch):
    monkeypatch.setattr(rag_tool, "_RETRY_DELAY_SECONDS", 0)  # don't slow the suite down for a real retry wait

    async def fake_embed_text(query, **kwargs):
        raise embeddings.EmbeddingError("HF is down", retriable=True)

    monkeypatch.setattr(rag_tool.embeddings, "embed_text", fake_embed_text)

    result = await rag_tool.search_company_knowledge("anything")

    assert "error" in result.lower()


async def test_search_retries_once_on_a_retriable_failure_then_succeeds(monkeypatch):
    """Real gap: EmbeddingError.retriable existed but nothing ever actually
    retried on it -- a purely transient HF hiccup (429 rate limit, 503
    model-loading) permanently failed the whole tool call mid-turn. Now one
    retry actually happens before giving up."""
    monkeypatch.setattr(rag_tool, "_RETRY_DELAY_SECONDS", 0)
    calls = []

    async def fake_embed_text(query, **kwargs):
        calls.append(query)
        if len(calls) == 1:
            raise embeddings.EmbeddingError("rate limited", retriable=True)
        return [0.1, 0.2]

    async def fake_search(query_vector, *, top_k):
        return [{"chunk_id": "x", "page_url": "u", "page_title": "T", "text": "content", "distance": 0.1}]

    monkeypatch.setattr(rag_tool.embeddings, "embed_text", fake_embed_text)
    monkeypatch.setattr(rag_tool.store, "search", fake_search)

    result = await rag_tool.search_company_knowledge("anything")

    assert len(calls) == 2  # the retry actually happened
    assert "content" in result


async def test_search_does_not_retry_a_non_retriable_failure(monkeypatch):
    calls = []

    async def fake_embed_text(query, **kwargs):
        calls.append(query)
        raise embeddings.EmbeddingError("HF_API_TOKEN not set", retriable=False)

    monkeypatch.setattr(rag_tool.embeddings, "embed_text", fake_embed_text)

    result = await rag_tool.search_company_knowledge("anything")

    assert len(calls) == 1  # no wasted retry on a config error that will never succeed
    assert "error" in result.lower()


def test_is_available_reflects_both_env_vars(monkeypatch):
    monkeypatch.delenv("HF_API_TOKEN", raising=False)
    monkeypatch.delenv("POSTGRES_DSN", raising=False)
    assert rag_tool.is_available() is False

    monkeypatch.setenv("HF_API_TOKEN", "test-token")
    assert rag_tool.is_available() is False  # still no POSTGRES_DSN

    monkeypatch.setenv("POSTGRES_DSN", "postgresql://fake/dsn")
    assert rag_tool.is_available() is True
