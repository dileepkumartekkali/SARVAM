"""End-to-end proof that the reasoning loop is genuinely agentic: a scripted
model requests a REAL tool, the REAL tool computes a real result, and the
loop feeds that real result back for the model to use — not a canned string,
an actual computed number the test doesn't know in advance until the tool
runs it for real.
"""

from agent_core.agents.task_agent import run_turn
from agent_core.llm_adapter import LLMRouter
from agent_core.supervisor.state import Mode, SessionState
from agent_core.tools import build_default_registry

from ._fakes import ScriptedProvider


def _session(mode=Mode.TEXT_TO_TEXT) -> SessionState:
    return SessionState(session_id="s", conversation_id="c", thread_id="t", mode=mode)


async def test_model_calls_real_calculator_tool_and_uses_the_real_result():
    registry = build_default_registry()
    provider = ScriptedProvider(
        [
            'TOOL_CALL: {"name": "calculate", "args": {"expression": "17 * 23"}}',
            "Based on the calculation, the answer is 391.",
            "OK",
        ]
    )
    router = LLMRouter([provider])

    result = await run_turn(
        _session(),
        router,
        "What's 17 times 23?",
        tools=registry.as_dispatch_dict(),
        tool_manifest=registry.as_prompt_manifest(),
    )

    assert result.tool_call_count == 1
    assert "391" in result.text  # the real, computed answer — 17*23 really is 391
    # Prove the tool's REAL output (not a canned mock) reached the model:
    # the wrapped tool result must contain the actual computed value.
    tool_result_message = provider.messages_by_call[1][-1]["content"]
    assert "391" in tool_result_message


async def test_model_uses_real_datetime_tool():
    registry = build_default_registry()
    provider = ScriptedProvider(
        [
            'TOOL_CALL: {"name": "get_current_datetime", "args": {"timezone": "UTC"}}',
            "Sure, I checked the clock for you.",
            "OK",
        ]
    )
    router = LLMRouter([provider])

    result = await run_turn(
        _session(), router, "What time is it?", tools=registry.as_dispatch_dict(), tool_manifest=registry.as_prompt_manifest()
    )

    assert result.tool_call_count == 1
    tool_result_message = provider.messages_by_call[1][-1]["content"]
    assert "UTC" in tool_result_message  # the real tz-aware timestamp, not a mock


async def test_system_prompt_actually_tells_the_model_the_tool_syntax():
    """The gap this session's answer flagged: previously nothing told the
    model TOOL_CALL existed at all. Confirm the manifest is really in the
    prompt the model receives."""
    registry = build_default_registry()
    provider = ScriptedProvider(["A plain answer, no tool needed.", "OK"])
    router = LLMRouter([provider])

    await run_turn(
        _session(), router, "hello", tools=registry.as_dispatch_dict(), tool_manifest=registry.as_prompt_manifest()
    )

    system_prompt_seen = provider.systems[0]  # call 0 is the main generation; call 1 is self-check
    assert "TOOL_CALL:" in system_prompt_seen
    assert "calculate(" in system_prompt_seen
    assert "delete_note(" in system_prompt_seen


async def test_write_scope_tool_from_real_registry_is_gated_in_voice_mode():
    """delete_note is marked write_scope in the real registry — confirm the
    Phase 6 confirmation gate actually engages for a real tool, not just the
    test-fake 'delete_account' used in Phase 6's own tests."""
    from agent_core.security.confirmation import ConfirmationGate

    registry = build_default_registry()
    gate = ConfirmationGate()
    provider = ScriptedProvider(['TOOL_CALL: {"name": "delete_note", "args": {"note_id": "1"}}', "unused"])
    router = LLMRouter([provider])

    result = await run_turn(
        _session(Mode.SPEECH_TO_SPEECH),
        router,
        "delete my note",
        tools=registry.as_dispatch_dict(),
        tool_manifest=registry.as_prompt_manifest(),
        write_scope_tools=registry.write_scope_names(),
        confirmation_gate=gate,
    )

    assert result.tool_call_count == 0  # never executed without confirmation
    assert result.pending_confirmation is not None
    assert result.pending_confirmation.tool_name == "delete_note"
