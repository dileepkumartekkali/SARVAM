"""Persistent chat history — a dedicated Postgres table (Supabase), separate
from LangGraph's `MemorySaver` checkpointer used in `supervisor/graph.py`.
That checkpointer stays capped/pruned (`_MAX_HISTORY_MESSAGES`) since it only
needs to hold enough context for the LLM's own reasoning; this module is the
permanent, user-facing chat log the frontend repopulates on refresh.

One ongoing conversation per user (not a multi-thread sidebar — see the
Supabase integration plan). Schema: `docs/supabase_schema.sql`.

Optional at runtime, same pattern as the checkpointer: no `POSTGRES_DSN` means
no persistence (every `/chat` call still works; history just doesn't survive
a restart), so local dev/CI/tests need no database.
"""

from __future__ import annotations

import os
import uuid

import asyncpg

_pool: asyncpg.Pool | None = None
_pool_dsn: str | None = None


async def _get_pool() -> asyncpg.Pool | None:
    global _pool, _pool_dsn
    dsn = os.environ.get("POSTGRES_DSN")
    if not dsn:
        return None
    if _pool is None or _pool_dsn != dsn:
        _pool = await asyncpg.create_pool(dsn, min_size=1, max_size=5)
        _pool_dsn = dsn
    return _pool


async def get_or_create_conversation(user_id: str) -> str | None:
    """Returns `None` (no-op) when persistence isn't configured."""
    pool = await _get_pool()
    if pool is None:
        return None
    uid = uuid.UUID(user_id)
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "select id from conversations where user_id = $1 order by created_at limit 1", uid
        )
        if row is not None:
            return str(row["id"])
        row = await conn.fetchrow("insert into conversations (user_id) values ($1) returning id", uid)
        return str(row["id"])


async def insert_message(
    conversation_id: str,
    user_id: str,
    role: str,
    content: str,
    response_language: str | None = None,
) -> str | None:
    """Returns `None` (no-op) when persistence isn't configured."""
    pool = await _get_pool()
    if pool is None:
        return None
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """insert into messages (conversation_id, user_id, role, content, response_language)
               values ($1, $2, $3, $4, $5) returning id""",
            uuid.UUID(conversation_id),
            uuid.UUID(user_id),
            role,
            content,
            response_language,
        )
        return str(row["id"])


async def list_messages(conversation_id: str, user_id: str) -> list[dict]:
    """Returns `[]` (no-op) when persistence isn't configured."""
    pool = await _get_pool()
    if pool is None:
        return []
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """select id, role, content, audio_path, response_language, created_at
               from messages where conversation_id = $1 and user_id = $2
               order by created_at""",
            uuid.UUID(conversation_id),
            uuid.UUID(user_id),
        )
        return [dict(r) for r in rows]
