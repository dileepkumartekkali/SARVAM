"""chat_store.py against a fake in-memory asyncpg pool — no live Postgres
needed. Covers: no-op when POSTGRES_DSN is unset (existing /chat tests rely
on this), and that conversation/message reads are scoped by user_id so one
user can never see another's history or conversations.
"""

import datetime
import uuid

import pytest

from agent_core.persistence import chat_store


class _FakeConnection:
    def __init__(self, conversations, messages):
        self._conversations = conversations
        self._messages = messages

    async def fetchrow(self, query, *args):
        if query.startswith("select id from conversations where id"):
            conversation_id, user_id = args
            row = self._conversations.get(conversation_id)
            return row if row and row["user_id"] == user_id else None
        if query.startswith("insert into conversations"):
            (user_id,) = args
            row = {"id": uuid.uuid4(), "user_id": user_id, "updated_at": datetime.datetime.now(datetime.timezone.utc)}
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
        if query.startswith("select id, role, content, audio_path, response_language"):
            conversation_id, user_id = args
            return [m for m in self._messages if m["conversation_id"] == conversation_id and m["user_id"] == user_id]
        if query.startswith("select c.id, c.updated_at"):
            (user_id,) = args
            rows = []
            for c in self._conversations.values():
                if c["user_id"] != user_id:
                    continue
                first = next((m for m in self._messages if m["conversation_id"] == c["id"]), None)
                rows.append({"id": c["id"], "updated_at": c["updated_at"], "title": first["content"] if first else None})
            return sorted(rows, key=lambda r: r["updated_at"], reverse=True)
        raise AssertionError(f"unexpected query: {query}")

    async def execute(self, query, *args):
        if query.startswith("update conversations set updated_at"):
            (conversation_id,) = args
            self._conversations[conversation_id]["updated_at"] = datetime.datetime.now(datetime.timezone.utc)
            return "UPDATE 1"
        if query.startswith("delete from conversations"):
            conversation_id, user_id = args
            row = self._conversations.get(conversation_id)
            if row is None or row["user_id"] != user_id:
                return "DELETE 0"
            del self._conversations[conversation_id]
            self._messages[:] = [m for m in self._messages if m["conversation_id"] != conversation_id]
            return "DELETE 1"
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
    assert chat_store.is_configured() is False
    assert await chat_store.create_conversation(str(uuid.uuid4())) is None
    assert await chat_store.get_conversation(str(uuid.uuid4()), str(uuid.uuid4())) is None
    assert await chat_store.list_conversations(str(uuid.uuid4())) == []
    assert await chat_store.list_messages(str(uuid.uuid4()), str(uuid.uuid4())) == []


async def test_create_conversation_then_get_conversation_by_owner(fake_pool):
    user_id = str(uuid.uuid4())

    conversation_id = await chat_store.create_conversation(user_id)

    assert await chat_store.get_conversation(conversation_id, user_id) is not None


async def test_get_conversation_rejects_non_owner(fake_pool):
    alice = str(uuid.uuid4())
    bob = str(uuid.uuid4())
    conversation_id = await chat_store.create_conversation(alice)

    # Same conversation id, wrong user — must not be visible to bob, even
    # though the backend's own connection would otherwise bypass RLS.
    assert await chat_store.get_conversation(conversation_id, bob) is None


async def test_multiple_conversations_are_isolated_and_ordered_by_recency(fake_pool):
    user_id = str(uuid.uuid4())
    first = await chat_store.create_conversation(user_id)
    second = await chat_store.create_conversation(user_id)  # created after "first"

    # Only "first" gets a message afterward — it should bump ahead of the more
    # recently *created* but not-yet-messaged "second" conversation.
    await chat_store.insert_message(first, user_id, "user", "first chat's message")

    first_messages = await chat_store.list_messages(first, user_id)
    second_messages = await chat_store.list_messages(second, user_id)
    assert [m["content"] for m in first_messages] == ["first chat's message"]
    assert second_messages == []  # each conversation's messages stay isolated

    conversations = await chat_store.list_conversations(user_id)
    assert [c["id"] for c in conversations] == [uuid.UUID(first), uuid.UUID(second)]
    assert conversations[0]["title"] == "first chat's message"
    assert conversations[1]["title"] is None  # never messaged -- no title yet


async def test_messages_are_scoped_to_owning_user(fake_pool):
    alice = str(uuid.uuid4())
    bob = str(uuid.uuid4())
    alice_conversation = await chat_store.create_conversation(alice)
    bob_conversation = await chat_store.create_conversation(bob)

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


async def test_delete_conversation_removes_it_and_its_messages(fake_pool):
    user_id = str(uuid.uuid4())
    conversation_id = await chat_store.create_conversation(user_id)
    await chat_store.insert_message(conversation_id, user_id, "user", "hello")

    assert await chat_store.delete_conversation(conversation_id, user_id) is True

    assert await chat_store.get_conversation(conversation_id, user_id) is None
    assert await chat_store.list_messages(conversation_id, user_id) == []


async def test_delete_conversation_rejects_non_owner(fake_pool):
    alice = str(uuid.uuid4())
    bob = str(uuid.uuid4())
    conversation_id = await chat_store.create_conversation(alice)

    assert await chat_store.delete_conversation(conversation_id, bob) is False
    # Untouched -- still there for the actual owner.
    assert await chat_store.get_conversation(conversation_id, alice) is not None


async def test_warm_up_is_a_no_op_without_a_dsn(monkeypatch):
    monkeypatch.delenv("POSTGRES_DSN", raising=False)
    await chat_store.warm_up()  # must not raise
