"""Proves task_agent's NATIVE tool-calling path (structured ToolCalls from
complete_with_tools, not the TOOL_CALL text convention) actually dispatches
real tools and continues the conversation with the correct message shape.
"""

from agent_core.agents.task_agent import run_turn
from agent_core.llm_adapter import LLMRouter
from agent_core.llm_adapter.base import ToolCall
from agent_core.supervisor.state import Mode, SessionState
from agent_core.tools import build_default_registry

from ._fakes import NativeToolCallProvider


def _session(mode=Mode.TEXT_TO_TEXT) -> SessionState:
    return SessionState(session_id="s", conversation_id="c", thread_id="t", mode=mode)


async def test_native_tool_call_dispatches_real_tool_and_uses_real_result():
    registry = build_default_registry()
    provider = NativeToolCallProvider(
        tool_calls_by_step=[[ToolCall(id="call_1", name="calculate", args={"expression": "17*23"})], []],
        final_texts=["", "17 times 23 is 391."],
    )
    router = LLMRouter([provider])

    result = await run_turn(
        _session(),
        router,
        "What's 17 times 23?",
        tools=registry.as_dispatch_dict(),
        tool_definitions=registry.as_tool_definitions(),
    )

    assert result.tool_call_count == 1
    assert "391" in result.text

    # The second call's messages must include a real "tool" role message
    # carrying the real computed result, correctly correlated by tool_call_id.
    second_call_messages = provider.messages_by_call[1]
    tool_messages = [m for m in second_call_messages if m.get("role") == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0]["tool_call_id"] == "call_1"
    assert "391" in tool_messages[0]["content"]


async def test_native_path_sends_real_json_schema_tool_definitions_to_provider():
    registry = build_default_registry()
    provider = NativeToolCallProvider(tool_calls_by_step=[[]], final_texts=["plain answer, no tool needed"])
    router = LLMRouter([provider])

    await run_turn(
        _session(),
        router,
        "hello",
        tools=registry.as_dispatch_dict(),
        tool_definitions=registry.as_tool_definitions(),
    )

    tools_sent = provider.tools_by_call[0]
    tool_names = {t.name for t in tools_sent}
    assert tool_names == {"calculate", "get_current_datetime", "convert_units", "save_note", "delete_note"}
    calculate_def = next(t for t in tools_sent if t.name == "calculate")
    assert calculate_def.parameters["expression"]["type"] == "string"
    assert "expression" in calculate_def.required


async def test_native_multiple_simultaneous_tool_calls_all_dispatch():
    """Real APIs can return several tool calls in one response — the loop
    must dispatch every one of them, not just the first."""
    registry = build_default_registry()
    provider = NativeToolCallProvider(
        tool_calls_by_step=[
            [
                ToolCall(id="call_1", name="calculate", args={"expression": "2+2"}),
                ToolCall(id="call_2", name="get_current_datetime", args={}),
            ],
            [],
        ],
        final_texts=["", "It's 4, and here's the time you asked for."],
    )
    router = LLMRouter([provider])

    result = await run_turn(
        _session(),
        router,
        "what's 2+2 and what time is it",
        tools=registry.as_dispatch_dict(),
        tool_definitions=registry.as_tool_definitions(),
    )

    assert result.tool_call_count == 2
    tool_messages = [m for m in provider.messages_by_call[1] if m.get("role") == "tool"]
    assert {m["tool_call_id"] for m in tool_messages} == {"call_1", "call_2"}


async def test_native_write_scope_tool_still_gated_by_confirmation():
    from agent_core.security.confirmation import ConfirmationGate

    registry = build_default_registry()
    gate = ConfirmationGate()
    provider = NativeToolCallProvider(
        tool_calls_by_step=[[ToolCall(id="call_1", name="delete_note", args={"note_id": "1"})]],
        final_texts=[""],
    )
    router = LLMRouter([provider])

    result = await run_turn(
        _session(Mode.SPEECH_TO_SPEECH),
        router,
        "delete my note",
        tools=registry.as_dispatch_dict(),
        tool_definitions=registry.as_tool_definitions(),
        write_scope_tools=registry.write_scope_names(),
        confirmation_gate=gate,
    )

    assert result.tool_call_count == 0
    assert result.pending_confirmation is not None
    assert result.pending_confirmation.tool_name == "delete_note"


async def test_native_tool_budget_still_enforced():
    provider = NativeToolCallProvider(
        tool_calls_by_step=[
            [ToolCall(id=f"call_{i}", name="noop", args={})] for i in range(10)
        ],
        final_texts=[""] * 10,
    )
    router = LLMRouter([provider])

    async def noop(**kwargs):
        return "ok"

    from agent_core.agents.task_agent import CLARIFYING_QUESTION

    result = await run_turn(
        _session(), router, "loop forever", tools={"noop": noop}, max_tool_calls=2
    )

    assert result.text == CLARIFYING_QUESTION
    assert result.tool_call_count == 2
