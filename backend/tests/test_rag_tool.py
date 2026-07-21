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


async def test_search_surfaces_embedding_failure_without_raising(monkeypatch):
    async def fake_embed_text(query, **kwargs):
        raise embeddings.EmbeddingError("HF is down", retriable=True)

    monkeypatch.setattr(rag_tool.embeddings, "embed_text", fake_embed_text)

    result = await rag_tool.search_company_knowledge("anything")

    assert "error" in result.lower()


def test_is_available_reflects_both_env_vars(monkeypatch):
    monkeypatch.delenv("HF_API_TOKEN", raising=False)
    monkeypatch.delenv("POSTGRES_DSN", raising=False)
    assert rag_tool.is_available() is False

    monkeypatch.setenv("HF_API_TOKEN", "test-token")
    assert rag_tool.is_available() is False  # still no POSTGRES_DSN

    monkeypatch.setenv("POSTGRES_DSN", "postgresql://fake/dsn")
    assert rag_tool.is_available() is True
