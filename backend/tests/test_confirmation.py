from agent_core.security.confirmation import ConfirmationGate


class FakeRedis:
    """Minimal in-memory stand-in for the redis.asyncio.Redis calls
    ConfirmationGate actually makes (get/set with a TTL, delete) -- no real
    Redis needed to test that path."""

    def __init__(self):
        self._store: dict[str, str] = {}

    async def set(self, key, value, ex=None):
        self._store[key] = value

    async def get(self, key):
        return self._store.get(key)

    async def delete(self, key):
        self._store.pop(key, None)


async def test_consume_succeeds_for_matching_tool_and_args():
    gate = ConfirmationGate()
    pending = await gate.request_confirmation("delete_account", {"account_id": "42"})

    assert await gate.consume(pending.token, "delete_account", {"account_id": "42"}) is True


async def test_consume_fails_for_different_args_same_tool():
    gate = ConfirmationGate()
    pending = await gate.request_confirmation("delete_account", {"account_id": "42"})

    assert await gate.consume(pending.token, "delete_account", {"account_id": "99"}) is False


async def test_consume_fails_for_different_tool_same_args():
    gate = ConfirmationGate()
    pending = await gate.request_confirmation("delete_account", {"account_id": "42"})

    assert await gate.consume(pending.token, "cancel_subscription", {"account_id": "42"}) is False


async def test_token_is_single_use():
    gate = ConfirmationGate()
    pending = await gate.request_confirmation("delete_account", {"account_id": "42"})

    assert await gate.consume(pending.token, "delete_account", {"account_id": "42"}) is True
    assert await gate.consume(pending.token, "delete_account", {"account_id": "42"}) is False  # replay rejected


async def test_unknown_token_rejected():
    gate = ConfirmationGate()
    assert await gate.consume("not-a-real-token", "delete_account", {"account_id": "42"}) is False


async def test_expired_token_is_rejected():
    """Real gap caught in a pre-deploy sweep: tokens used to never expire
    at all, growing unbounded in this process-lifetime singleton and
    staying exploitable indefinitely if one ever leaked. ttl_seconds=-1
    means it's already expired the instant it's issued -- deterministic,
    no time mocking needed."""
    gate = ConfirmationGate(ttl_seconds=-1)
    pending = await gate.request_confirmation("delete_account", {"account_id": "42"})

    assert await gate.consume(pending.token, "delete_account", {"account_id": "42"}) is False


async def test_expired_tokens_are_evicted_on_next_request():
    gate = ConfirmationGate(ttl_seconds=-1)
    await gate.request_confirmation("delete_account", {"account_id": "42"})
    assert len(gate._pending) == 1

    await gate.request_confirmation("cancel_subscription", {"account_id": "99"})

    # The first (already-expired) entry was swept, not left to accumulate.
    assert len(gate._pending) == 1


async def test_redis_backed_path_enforces_the_same_single_use_semantics():
    """Architecture gap closed: tokens used to be in-memory/single-process
    only -- a confirmation issued on one instance was unredeemable on
    another. With REDIS_URL configured, the same exact-match, single-use,
    replay-rejected semantics are enforced through Redis instead."""
    gate = ConfirmationGate()
    gate._redis = FakeRedis()
    gate._redis_checked = True  # skip the real REDIS_URL/from_url lookup

    pending = await gate.request_confirmation("delete_account", {"account_id": "42"})

    assert await gate.consume(pending.token, "delete_account", {"account_id": "99"}) is False  # wrong args
    assert await gate.consume(pending.token, "cancel_subscription", {"account_id": "42"}) is False  # wrong tool
    assert await gate.consume(pending.token, "delete_account", {"account_id": "42"}) is True
    assert await gate.consume(pending.token, "delete_account", {"account_id": "42"}) is False  # replay rejected
    assert await gate.consume("not-a-real-token", "delete_account", {"account_id": "42"}) is False
