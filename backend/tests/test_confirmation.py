from agent_core.security.confirmation import ConfirmationGate


def test_consume_succeeds_for_matching_tool_and_args():
    gate = ConfirmationGate()
    pending = gate.request_confirmation("delete_account", {"account_id": "42"})

    assert gate.consume(pending.token, "delete_account", {"account_id": "42"}) is True


def test_consume_fails_for_different_args_same_tool():
    gate = ConfirmationGate()
    pending = gate.request_confirmation("delete_account", {"account_id": "42"})

    assert gate.consume(pending.token, "delete_account", {"account_id": "99"}) is False


def test_consume_fails_for_different_tool_same_args():
    gate = ConfirmationGate()
    pending = gate.request_confirmation("delete_account", {"account_id": "42"})

    assert gate.consume(pending.token, "cancel_subscription", {"account_id": "42"}) is False


def test_token_is_single_use():
    gate = ConfirmationGate()
    pending = gate.request_confirmation("delete_account", {"account_id": "42"})

    assert gate.consume(pending.token, "delete_account", {"account_id": "42"}) is True
    assert gate.consume(pending.token, "delete_account", {"account_id": "42"}) is False  # replay rejected


def test_unknown_token_rejected():
    gate = ConfirmationGate()
    assert gate.consume("not-a-real-token", "delete_account", {"account_id": "42"}) is False


def test_expired_token_is_rejected():
    """Real gap caught in a pre-deploy sweep: tokens used to never expire
    at all, growing unbounded in this process-lifetime singleton and
    staying exploitable indefinitely if one ever leaked. ttl_seconds=-1
    means it's already expired the instant it's issued -- deterministic,
    no time mocking needed."""
    gate = ConfirmationGate(ttl_seconds=-1)
    pending = gate.request_confirmation("delete_account", {"account_id": "42"})

    assert gate.consume(pending.token, "delete_account", {"account_id": "42"}) is False


def test_expired_tokens_are_evicted_on_next_request():
    gate = ConfirmationGate(ttl_seconds=-1)
    gate.request_confirmation("delete_account", {"account_id": "42"})
    assert len(gate._pending) == 1

    gate.request_confirmation("cancel_subscription", {"account_id": "99"})

    # The first (already-expired) entry was swept, not left to accumulate.
    assert len(gate._pending) == 1
