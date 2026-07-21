"""agent_core.rag.store against a fake in-memory asyncpg pool -- same
pattern as tests/test_chat_store.py. The fake computes real cosine distance
so ordering assertions mean something, not just "a query ran."
"""

import math

import pytest

from agent_core.rag import store


def _cosine_distance(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return 1 - dot / (norm_a * norm_b)


def _parse_vector_literal(literal: str) -> list[float]:
    return [float(x) for x in literal.strip("[]").split(",")]


class _FakeConnection:
    def __init__(self, rows: dict):
        self._rows = rows  # chunk_id -> row dict

    async def execute(self, query, *args):
        if query.startswith("insert into rag_chunks"):
            chunk_id, page_url, page_title, text, embedding_literal = args
            self._rows[chunk_id] = {
                "chunk_id": chunk_id,
                "page_url": page_url,
                "page_title": page_title,
                "text": text,
                "embedding": _parse_vector_literal(embedding_literal),
            }
            return "INSERT 1"
        raise AssertionError(f"unexpected execute: {query}")

    async def fetch(self, query, *args):
        if query.startswith("select chunk_id, page_url, page_title, text"):
            embedding_literal, top_k = args
            query_vector = _parse_vector_literal(embedding_literal)
            scored = [
                {**{k: v for k, v in r.items() if k != "embedding"}, "distance": _cosine_distance(query_vector, r["embedding"])}
                for r in self._rows.values()
            ]
            return sorted(scored, key=lambda r: r["distance"])[:top_k]
        raise AssertionError(f"unexpected query: {query}")


class _FakeAcquireCtx:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *exc):
        return False


class _FakePool:
    def __init__(self):
        self._rows = {}

    def acquire(self):
        return _FakeAcquireCtx(_FakeConnection(self._rows))


@pytest.fixture
def fake_pool(monkeypatch):
    pool = _FakePool()

    async def fake_create_pool(dsn, **kwargs):
        return pool

    monkeypatch.setenv("POSTGRES_DSN", "postgresql://fake/dsn")
    monkeypatch.setattr(store, "_pool", None)
    monkeypatch.setattr(store, "_pool_dsn", None)
    monkeypatch.setattr(store.asyncpg, "create_pool", fake_create_pool)
    return pool


async def test_no_dsn_is_a_silent_no_op(monkeypatch):
    monkeypatch.delenv("POSTGRES_DSN", raising=False)
    assert store.is_configured() is False
    assert await store.upsert_chunk("id1", "url", "title", "text", [0.1, 0.2]) is None
    assert await store.search([0.1, 0.2]) == []


async def test_search_returns_closest_match_first(fake_pool):
    await store.upsert_chunk("about#0", "https://x/about", "About", "we build software", [1.0, 0.0])
    await store.upsert_chunk("blog#0", "https://x/blog", "Blog", "unrelated post", [0.0, 1.0])

    results = await store.search([0.9, 0.1], top_k=5)

    assert results[0]["chunk_id"] == "about#0"
    assert results[1]["chunk_id"] == "blog#0"
    assert results[0]["distance"] < results[1]["distance"]


async def test_search_respects_top_k(fake_pool):
    for i in range(5):
        await store.upsert_chunk(f"c{i}", "https://x", "title", "text", [1.0, float(i)])

    results = await store.search([1.0, 0.0], top_k=2)

    assert len(results) == 2


async def test_upsert_is_idempotent_by_chunk_id(fake_pool):
    await store.upsert_chunk("about#0", "https://x/about", "About", "old text", [1.0, 0.0])
    await store.upsert_chunk("about#0", "https://x/about", "About", "new text", [1.0, 0.0])

    results = await store.search([1.0, 0.0], top_k=5)

    assert len(results) == 1
    assert results[0]["text"] == "new text"
