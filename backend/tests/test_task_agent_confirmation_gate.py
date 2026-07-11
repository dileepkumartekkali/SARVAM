"""Voice write-scope confirmation is a hard gate in run_turn's tool dispatch,
not a prompt instruction — these tests drive it directly, without a real LLM."""

from agent_core.agents.task_agent import CONFIRMATION_REQUIRED_TEXT, run_turn
from agent_core.llm_adapter import LLMRouter
from agent_core.security.confirmation import ConfirmationGate
from agent_core.supervisor.state import Mode, SessionState

from ._fakes import ScriptedProvider

DELETE_CALL = 'TOOL_CALL: {"name": "delete_account", "args": {"account_id": "42"}}'


async def delete_account(**kwargs) -> str:
    return "account deleted"


def _voice_session() -> SessionState:
    return SessionState(session_id="s", conversation_id="c", thread_id="t", mode=Mode.SPEECH_TO_SPEECH)


def _text_session() -> SessionState:
    return SessionState(session_id="s", conversation_id="c", thread_id="t", mode=Mode.TEXT_TO_TEXT)


async def test_write_scope_tool_in_voice_mode_is_blocked_without_confirmation():
    provider = ScriptedProvider([DELETE_CALL, "unused"])
    router = LLMRouter([provider])
    gate = ConfirmationGate()

    result = await run_turn(
        _voice_session(),
        router,
        "delete my account",
        tools={"delete_account": delete_account},
        write_scope_tools={"delete_account"},
        confirmation_gate=gate,
    )

    assert result.text == CONFIRMATION_REQUIRED_TEXT
    assert result.tool_call_count == 0  # never executed
    assert result.pending_confirmation is not None
    assert result.pending_confirmation.tool_name == "delete_account"


async def test_write_scope_tool_executes_once_confirmation_token_matches():
    gate = ConfirmationGate()
    pending = gate.request_confirmation("delete_account", {"account_id": "42"})
    provider = ScriptedProvider([DELETE_CALL, "Done, your account has been deleted.", "OK"])
    router = LLMRouter([provider])

    result = await run_turn(
        _voice_session(),
        router,
        "yes, confirm",
        tools={"delete_account": delete_account},
        write_scope_tools={"delete_account"},
        confirmation_gate=gate,
        confirmation_token=pending.token,
    )

    assert result.text == "Done, your account has been deleted."
    assert result.tool_call_count == 1


async def test_mismatched_confirmation_token_is_rejected_not_executed():
    gate = ConfirmationGate()
    pending = gate.request_confirmation("delete_account", {"account_id": "99"})  # different args
    provider = ScriptedProvider([DELETE_CALL, "unused"])
    router = LLMRouter([provider])

    result = await run_turn(
        _voice_session(),
        router,
        "delete my account",
        tools={"delete_account": delete_account},
        write_scope_tools={"delete_account"},
        confirmation_gate=gate,
        confirmation_token=pending.token,
    )

    assert result.text == CONFIRMATION_REQUIRED_TEXT
    assert result.tool_call_count == 0


async def test_write_scope_tool_in_text_mode_does_not_require_confirmation():
    """The gate is scoped to voice per the requirement — typed confirmation
    exists precisely because voice lacks the friction a text UI already has."""
    provider = ScriptedProvider([DELETE_CALL, "Deleted.", "OK"])
    router = LLMRouter([provider])

    result = await run_turn(
        _text_session(),
        router,
        "delete my account",
        tools={"delete_account": delete_account},
        write_scope_tools={"delete_account"},
    )

    assert result.text == "Deleted."
    assert result.tool_call_count == 1


async def test_replayed_token_cannot_be_reused_for_a_second_execution():
    gate = ConfirmationGate()
    pending = gate.request_confirmation("delete_account", {"account_id": "42"})
    provider = ScriptedProvider([DELETE_CALL, "Done.", "OK", DELETE_CALL, "unused"])
    router = LLMRouter([provider])

    first = await run_turn(
        _voice_session(),
        router,
        "confirm",
        tools={"delete_account": delete_account},
        write_scope_tools={"delete_account"},
        confirmation_gate=gate,
        confirmation_token=pending.token,
    )
    assert first.tool_call_count == 1

    second = await run_turn(
        _voice_session(),
        router,
        "delete my account again",
        tools={"delete_account": delete_account},
        write_scope_tools={"delete_account"},
        confirmation_gate=gate,
        confirmation_token=pending.token,  # same token, replayed
    )

    assert second.text == CONFIRMATION_REQUIRED_TEXT
    assert second.tool_call_count == 0
