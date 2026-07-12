"""chat_store.py against a fake in-memory asyncpg pool — no live Postgres
needed. Covers: no-op when POSTGRES_DSN is unset (existing /chat tests rely
on this), and that conversation/message reads are scoped by user_id so one
user can never see another's history.
"""

import uuid

import pytest

from agent_core.persistence import chat_store


class _FakeConnection:
    def __init__(self, conversations, messages):
        self._conversations = conversations
        self._messages = messages

    async def fetchrow(self, query, *args):
        if query.startswith("select id from conversations"):
            (user_id,) = args
            matches = [c for c in self._conversations.values() if c["user_id"] == user_id]
            return matches[0] if matches else None
        if query.startswith("insert into conversations"):
            (user_id,) = args
            row = {"id": uuid.uuid4(), "user_id": user_id}
            self._conversations[row["id"]] = row
            return row
        if query.startswith("insert into messages"):
            conversation_id, user_id, role, content, response_language = args
            row = {
                "id": uuid.uuid4(),
                "conversation_id": conversation_id,
                "user_id": user_id,
                "role": role,
                "content": content,
                "audio_path": None,
                "response_language": response_language,
            }
            self._messages.append(row)
            return row
        raise AssertionError(f"unexpected query: {query}")

    async def fetch(self, query, *args):
        assert query.startswith("select id, role, content, audio_path, response_language")
        conversation_id, user_id = args
        return [m for m in self._messages if m["conversation_id"] == conversation_id and m["user_id"] == user_id]

    async def execute(self, query, *args):
        raise AssertionError(f"unexpected execute: {query}")


class _FakeAcquireCtx:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *exc):
        return False


class _FakePool:
    def __init__(self):
        self._conversations = {}
        self._messages = []

    def acquire(self):
        return _FakeAcquireCtx(_FakeConnection(self._conversations, self._messages))


@pytest.fixture
def fake_pool(monkeypatch):
    pool = _FakePool()

    async def fake_create_pool(dsn, **kwargs):
        return pool

    monkeypatch.setenv("POSTGRES_DSN", "postgresql://fake/dsn")
    monkeypatch.setattr(chat_store, "_pool", None)
    monkeypatch.setattr(chat_store, "_pool_dsn", None)
    monkeypatch.setattr(chat_store.asyncpg, "create_pool", fake_create_pool)
    return pool


async def test_no_dsn_is_a_silent_no_op(monkeypatch):
    monkeypatch.delenv("POSTGRES_DSN", raising=False)
    assert await chat_store.get_or_create_conversation(str(uuid.uuid4())) is None
    assert await chat_store.list_messages(str(uuid.uuid4()), str(uuid.uuid4())) == []


async def test_get_or_create_conversation_is_idempotent_per_user(fake_pool):
    user_id = str(uuid.uuid4())

    first = await chat_store.get_or_create_conversation(user_id)
    second = await chat_store.get_or_create_conversation(user_id)

    assert first == second


async def test_messages_are_scoped_to_owning_user(fake_pool):
    alice = str(uuid.uuid4())
    bob = str(uuid.uuid4())
    alice_conversation = await chat_store.get_or_create_conversation(alice)
    bob_conversation = await chat_store.get_or_create_conversation(bob)

    await chat_store.insert_message(alice_conversation, alice, "user", "alice's message")
    await chat_store.insert_message(bob_conversation, bob, "user", "bob's message")

    alice_messages = await chat_store.list_messages(alice_conversation, alice)
    bob_messages = await chat_store.list_messages(bob_conversation, bob)

    assert [m["content"] for m in alice_messages] == ["alice's message"]
    assert [m["content"] for m in bob_messages] == ["bob's message"]

    # Even with the right conversation id, the wrong user_id sees nothing —
    # this is the ownership check that matters since the backend's own
    # connection bypasses RLS (trusted server-side query, not PostgREST).
    assert await chat_store.list_messages(alice_conversation, bob) == []
