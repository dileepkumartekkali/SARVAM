"""pgvector-backed chunk storage -- the SAME Supabase Postgres already used
by agent_core/persistence/chat_store.py, not a separate vector database.
One extension (`vector`), one new table; no new service to run or pay for.

Embeddings are formatted as pgvector's text literal ("[0.1,0.2,...]") and
cast with `::vector` in SQL rather than pulling in the separate `pgvector`
pip package's asyncpg codec -- similarity search happens entirely in SQL
(the `<=>` cosine-distance operator), so Python never needs to parse a
vector value back out, only send one.
"""

from __future__ import annotations

import os

import asyncpg

_pool: asyncpg.Pool | None = None
_pool_dsn: str | None = None


def is_configured() -> bool:
    return bool(os.environ.get("POSTGRES_DSN"))


async def _get_pool() -> asyncpg.Pool | None:
    global _pool, _pool_dsn
    dsn = os.environ.get("POSTGRES_DSN")
    if not dsn:
        return None
    if _pool is None or _pool_dsn != dsn:
        # statement_cache_size=0 -- same reason as chat_store.py: Supabase's
        # PgBouncer (transaction mode) doesn't support asyncpg's default
        # server-side prepared statements.
        _pool = await asyncpg.create_pool(dsn, min_size=1, max_size=5, statement_cache_size=0)
        _pool_dsn = dsn
    return _pool


def _to_vector_literal(embedding: list[float]) -> str:
    return "[" + ",".join(repr(x) for x in embedding) + "]"


async def upsert_chunk(chunk_id: str, page_url: str, page_title: str, text: str, embedding: list[float]) -> None:
    """Idempotent by chunk_id -- re-running ingestion after a re-scrape
    updates existing rows in place instead of accumulating duplicates."""
    pool = await _get_pool()
    if pool is None:
        return
    async with pool.acquire() as conn:
        await conn.execute(
            """insert into rag_chunks (chunk_id, page_url, page_title, text, embedding)
               values ($1, $2, $3, $4, $5::vector)
               on conflict (chunk_id) do update
                 set page_url = excluded.page_url, page_title = excluded.page_title,
                     text = excluded.text, embedding = excluded.embedding""",
            chunk_id,
            page_url,
            page_title,
            text,
            _to_vector_literal(embedding),
        )


async def search(query_embedding: list[float], *, top_k: int = 5) -> list[dict]:
    """Nearest neighbors by cosine distance (`<=>`, pgvector's operator --
    lower is more similar). Returns `[]` (no-op) when persistence isn't
    configured, same convention as chat_store.py, so the RAG tool degrades
    to "no results" rather than erroring when the feature isn't set up."""
    pool = await _get_pool()
    if pool is None:
        return []
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """select chunk_id, page_url, page_title, text,
                      embedding <=> $1::vector as distance
               from rag_chunks
               order by distance asc
               limit $2""",
            _to_vector_literal(query_embedding),
            top_k,
        )
        return [dict(r) for r in rows]
