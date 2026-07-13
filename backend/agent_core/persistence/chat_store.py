"""Persistent chat history — a dedicated Postgres table (Supabase), separate
from LangGraph's `MemorySaver` checkpointer used in `supervisor/graph.py`.
That checkpointer stays capped/pruned (`_MAX_HISTORY_MESSAGES`) since it only
needs to hold enough context for the LLM's own reasoning; this module is the
permanent, user-facing chat log the frontend repopulates on refresh.

Multiple conversations per user (ChatGPT/Claude-style switching) — every
query is scoped by `user_id` so one user's conversations/messages are never
visible to another. Schema: `docs/supabase_schema.sql` (unchanged — no new
table/column needed for multi-conversation support).

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


def is_configured() -> bool:
    """Whether persistence is wired up at all — callers use this to tell
    apart "no DB configured, skip the check" from "checked and not found,
    404 it" (both otherwise look like a `None` return from get_conversation)."""
    return bool(os.environ.get("POSTGRES_DSN"))


async def _get_pool() -> asyncpg.Pool | None:
    global _pool, _pool_dsn
    dsn = os.environ.get("POSTGRES_DSN")
    if not dsn:
        return None
    if _pool is None or _pool_dsn != dsn:
        # Supabase's connection pooler (PgBouncer, transaction mode) doesn't
        # support server-side prepared statements the way asyncpg uses by
        # default -- every query fails with a "prepared statement ... does
        # not exist"/duplicate error. statement_cache_size=0 turns that off.
        _pool = await asyncpg.create_pool(dsn, min_size=1, max_size=5, statement_cache_size=0)
        _pool_dsn = dsn
    return _pool


async def warm_up() -> None:
    """Pre-creates the connection pool at app startup instead of paying that
    connect cost on a user's very first request after a cold start (Render
    free-tier idles the whole process, not just the DB connection)."""
    await _get_pool()


async def delete_conversation(conversation_id: str, user_id: str) -> bool:
    """Ownership-scoped delete — `messages` rows cascade-delete via the FK
    (schema: `on delete cascade`), no separate cleanup needed. Returns
    whether a row was actually deleted (false = didn't exist or wasn't
    yours; the caller reports both as 404, never which)."""
    pool = await _get_pool()
    if pool is None:
        return False
    async with pool.acquire() as conn:
        result = await conn.execute(
            "delete from conversations where id = $1 and user_id = $2",
            uuid.UUID(conversation_id),
            uuid.UUID(user_id),
        )
        return result == "DELETE 1"


async def list_conversations(user_id: str) -> list[dict]:
    """Returns `[]` (no-op) when persistence isn't configured. Ordered most-
    recently-active first; `title` is derived from the first message (no
    dedicated column) and is `None` for a brand-new, still-empty conversation."""
    pool = await _get_pool()
    if pool is None:
        return []
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """select c.id, c.updated_at,
                      (select content from messages m
                       where m.conversation_id = c.id order by m.created_at limit 1) as title
               from conversations c where c.user_id = $1
               order by c.updated_at desc""",
            uuid.UUID(user_id),
        )
        return [dict(r) for r in rows]


async def create_conversation(user_id: str) -> str | None:
    """Returns `None` (no-op) when persistence isn't configured."""
    pool = await _get_pool()
    if pool is None:
        return None
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "insert into conversations (user_id) values ($1) returning id", uuid.UUID(user_id)
        )
        return str(row["id"])


async def get_conversation(conversation_id: str, user_id: str) -> dict | None:
    """Ownership check — `None` means "doesn't exist or isn't yours," both
    correctly surfaced as a 404 by the caller (never leaks which). Returns
    `None` (no-op, not an error) when persistence isn't configured."""
    pool = await _get_pool()
    if pool is None:
        return None
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "select id from conversations where id = $1 and user_id = $2",
            uuid.UUID(conversation_id),
            uuid.UUID(user_id),
        )
        return dict(row) if row is not None else None


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
        # Bumps the conversation to the top of list_conversations' ordering.
        await conn.execute(
            "update conversations set updated_at = now() where id = $1", uuid.UUID(conversation_id)
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
