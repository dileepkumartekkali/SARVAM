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

import asyncio
import logging
import os
import uuid

import asyncpg

logger = logging.getLogger("agent_core.persistence")

_pool: asyncpg.Pool | None = None
_pool_dsn: str | None = None


def is_configured() -> bool:
    """Whether persistence is wired up at all — callers use this to tell
    apart "no DB configured, skip the check" from "checked and not found,
    404 it" (both otherwise look like a `None` return from get_conversation)."""
    return bool(os.environ.get("POSTGRES_DSN"))


def _client_supplied_uuid(value: str) -> uuid.UUID | None:
    """`conversation_id` (unlike `user_id`, which is always the server-
    verified Supabase subject) is client-supplied and never validated as a
    real UUID before reaching here. Real gap: `uuid.UUID(conversation_id)`
    raised a bare `ValueError` on anything malformed, uncaught anywhere on
    the path -- a 500 instead of the 404 these endpoints are clearly built
    to return for "doesn't exist or isn't yours". Returns `None` on a
    malformed id so callers can treat it exactly like "not found"."""
    try:
        return uuid.UUID(value)
    except ValueError:
        return None


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
    conv_uuid = _client_supplied_uuid(conversation_id)
    if conv_uuid is None:
        return False
    pool = await _get_pool()
    if pool is None:
        return False
    async with pool.acquire() as conn:
        result = await conn.execute(
            "delete from conversations where id = $1 and user_id = $2",
            conv_uuid,
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
    conv_uuid = _client_supplied_uuid(conversation_id)
    if conv_uuid is None:
        return None
    pool = await _get_pool()
    if pool is None:
        return None
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "select id from conversations where id = $1 and user_id = $2",
            conv_uuid,
            uuid.UUID(user_id),
        )
        return dict(row) if row is not None else None


async def _insert_message_on(
    conn,
    conversation_id: str,
    user_id: str,
    role: str,
    content: str,
    response_language: str | None = None,
    message_id: str | None = None,
) -> str:
    """The actual insert, run on a caller-supplied connection so `record_turn`
    can wrap two calls in one transaction. `insert_message` (below) is the
    standalone version most callers want; this is the shared plumbing."""
    if message_id is not None:
        row = await conn.fetchrow(
            """insert into messages (id, conversation_id, user_id, role, content, response_language)
               values ($1, $2, $3, $4, $5, $6) returning id""",
            uuid.UUID(message_id),
            uuid.UUID(conversation_id),
            uuid.UUID(user_id),
            role,
            content,
            response_language,
        )
    else:
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
    await conn.execute("update conversations set updated_at = now() where id = $1", uuid.UUID(conversation_id))
    return str(row["id"])


async def insert_message(
    conversation_id: str,
    user_id: str,
    role: str,
    content: str,
    response_language: str | None = None,
    message_id: str | None = None,
) -> str | None:
    """Returns `None` (no-op) when persistence isn't configured. `message_id`
    lets a caller pre-generate the id (see `record_turn`) so it can hand the
    id back to the client before the row actually exists."""
    pool = await _get_pool()
    if pool is None:
        return None
    async with pool.acquire() as conn:
        return await _insert_message_on(conn, conversation_id, user_id, role, content, response_language, message_id)


_RECORD_TURN_MAX_ATTEMPTS = 3
_RECORD_TURN_RETRY_DELAY_SECONDS = 0.5


async def record_turn(
    conversation_id: str,
    user_id: str,
    user_message: str,
    assistant_message: str,
    response_language: str | None,
    assistant_message_id: str,
) -> None:
    """Persists both sides of a turn — meant to run as a FastAPI background
    task, AFTER the HTTP response already went out, so a user never waits on
    a DB write just to see the answer they were given synchronously. No-op
    if persistence isn't configured.

    Both inserts run in ONE transaction (never a half-saved turn — the user
    message without its reply, or vice versa) and retry up to 3 times with a
    short delay to ride out a transient blip (a dropped connection, a
    momentary Supabase hiccup) rather than losing the turn to something that
    would have worked a second later. If every attempt fails, the failure is
    logged, not raised — a lost history write must never surface as an error
    for a turn the user already got a real answer to (see the accepted
    remaining risk — process death mid-task — in docs/SAD.md)."""
    pool = await _get_pool()
    if pool is None:
        return
    for attempt in range(_RECORD_TURN_MAX_ATTEMPTS):
        try:
            async with pool.acquire() as conn:
                async with conn.transaction():
                    await _insert_message_on(conn, conversation_id, user_id, "user", user_message)
                    await _insert_message_on(
                        conn, conversation_id, user_id, "assistant", assistant_message, response_language,
                        message_id=assistant_message_id,
                    )
            return
        except Exception:
            if attempt == _RECORD_TURN_MAX_ATTEMPTS - 1:
                logger.exception(
                    "record_turn failed after %d attempts for conversation_id=%s -- turn was answered but not persisted",
                    _RECORD_TURN_MAX_ATTEMPTS,
                    conversation_id,
                )
            else:
                await asyncio.sleep(_RECORD_TURN_RETRY_DELAY_SECONDS * (attempt + 1))


async def list_messages(conversation_id: str, user_id: str) -> list[dict]:
    """Returns `[]` (no-op) when persistence isn't configured."""
    conv_uuid = _client_supplied_uuid(conversation_id)
    if conv_uuid is None:
        return []
    pool = await _get_pool()
    if pool is None:
        return []
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """select id, role, content, audio_path, response_language, created_at
               from messages where conversation_id = $1 and user_id = $2
               order by created_at""",
            conv_uuid,
            uuid.UUID(user_id),
        )
        return [dict(r) for r in rows]
