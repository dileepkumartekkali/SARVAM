"""Proves the cancellation token threaded through run_turn's tool loop
actually aborts an in-flight LLM call, and never leaves a tool call
half-executed — a dispatched tool either finishes before cancellation is
noticed, or never starts."""

import asyncio

from agent_core.agents.cancellation import CancellationToken
from agent_core.agents.task_agent import run_turn
from agent_core.llm_adapter import LLMRouter
from agent_core.llm_adapter.base import CompletionResult
from agent_core.supervisor.state import Mode, SessionState

from ._fakes import ScriptedProvider


class SlowProvider:
    """Never returns on its own — only a cancellation can end the call."""

    name = "slow"

    async def complete(self, messages, *, system=None, max_tokens=None, temperature=None):
        await asyncio.sleep(30)
        return "should never get here"

    async def stream(self, messages, *, system=None, max_tokens=None, temperature=None):
        await asyncio.sleep(30)
        yield "should never get here"

    async def complete_with_tools(self, messages, *, system=None, tools=None, max_tokens=None, temperature=None):
        await asyncio.sleep(30)
        return CompletionResult(text="should never get here")


def _session() -> SessionState:
    return SessionState(session_id="s", conversation_id="c", thread_id="t", mode=Mode.TEXT_TO_TEXT)


async def test_cancellation_aborts_an_in_flight_llm_call():
    token = CancellationToken()
    router = LLMRouter([SlowProvider()])

    async def cancel_soon():
        await asyncio.sleep(0.01)
        token.cancel()

    asyncio.ensure_future(cancel_soon())
    result = await run_turn(_session(), router, "hello", cancellation_token=token)

    assert result.cancelled is True
    assert result.text == ""


async def test_tool_call_completes_fully_before_cancellation_is_honored():
    """Cancel fires while a tool is mid-execution — the tool must still run
    to completion and its result must be recorded; only the NEXT step (the
    next LLM call) is what actually gets cancelled."""
    token = CancellationToken()
    tool_finished = False

    async def slow_tool(**kwargs):
        nonlocal tool_finished
        await asyncio.sleep(0.05)
        tool_finished = True
        return "tool result"

    provider = ScriptedProvider(
        ['TOOL_CALL: {"name": "slow_tool", "args": {}}']  # only one scripted reply — loop must cancel, not repeat
    )
    router = LLMRouter([provider])

    async def cancel_during_tool_execution():
        await asyncio.sleep(0.02)  # fires while slow_tool is still running
        token.cancel()

    asyncio.ensure_future(cancel_during_tool_execution())
    result = await run_turn(
        _session(), router, "hello", tools={"slow_tool": slow_tool}, cancellation_token=token
    )

    assert tool_finished is True  # the dispatched tool call ran to completion
    assert result.cancelled is True  # but the loop stopped at the next step, not mid-tool
