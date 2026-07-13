"""stream_turn — the streaming counterpart to run_turn. Covers: plain-reply
happy path (chunks stream through, self-check's reviewer call still runs),
falling back to run_turn on a sniffed TOOL_CALL: prefix (nothing streamed
was ever wrong since nothing was yielded before the fallback), and a
self-check failure being logged rather than corrected (no regeneration call
— the accepted trade-off vs. run_turn's correction retry)."""

from agent_core.agents.task_agent import stream_turn
from agent_core.llm_adapter import LLMRouter
from agent_core.supervisor.state import Mode, SessionState

from ._fakes import ScriptedProvider


def _session(mode=Mode.TEXT_TO_TEXT) -> SessionState:
    return SessionState(session_id="s", conversation_id="c", thread_id="t", mode=mode)


async def noop_tool(**kwargs) -> str:
    return "ok"


async def test_stream_turn_yields_text_deltas_then_done_on_plain_reply():
    provider = ScriptedProvider(["Hello there. How are you today?", "OK"])
    router = LLMRouter([provider])

    events = [e async for e in stream_turn(_session(), router, "hi")]

    text_deltas = [e["text"] for e in events if e["type"] == "text_delta"]
    combined = " ".join(text_deltas)
    assert "Hello" in combined and "today" in combined

    done = events[-1]
    assert done["type"] == "done"
    assert done["self_check_ok"] is True
    assert done["tool_call_count"] == 0
    assert done["pending_confirmation"] is None
    # Both the streamed reply AND self-check's separate reviewer call happened.
    assert provider.call_count == 2


async def test_stream_turn_falls_back_to_run_turn_on_tool_call_prefix():
    """The sniffed-and-discarded first call, then run_turn's own dispatch
    call, its final-answer call, and self-check's reviewer call — four calls
    total, none of which ever streamed the raw `TOOL_CALL:` text to anyone."""
    provider = ScriptedProvider(
        [
            'TOOL_CALL: {"name": "noop", "args": {}}',  # sniffed by stream_turn, discarded
            'TOOL_CALL: {"name": "noop", "args": {}}',  # run_turn's own first call
            "done",  # run_turn's final-answer call after the tool result
            "OK",  # self-check's reviewer call
        ]
    )
    router = LLMRouter([provider])

    events = [
        e async for e in stream_turn(_session(), router, "what time is it", tools={"noop": noop_tool})
    ]

    # Nothing resembling the raw tool-call text was ever yielded as a delta.
    for e in events:
        if e["type"] == "text_delta":
            assert "TOOL_CALL" not in e["text"]

    done = events[-1]
    assert done["type"] == "done"
    assert done["text"] == "done"
    assert done["tool_call_count"] == 1
    assert done["self_check_ok"] is True


async def test_stream_turn_self_check_failure_is_logged_not_corrected():
    long_reply = " ".join(["word"] * 200)  # exceeds TEXT_TO_TEXT's 160-word cap -- a deterministic check
    provider = ScriptedProvider([long_reply])
    router = LLMRouter([provider])

    events = [e async for e in stream_turn(_session(), router, "hi")]

    done = events[-1]
    assert done["self_check_ok"] is False
    assert "length violation" in done["self_check_reason"]
    # Deterministic violations short-circuit before the reviewer LLM call, and
    # there is no correction/regeneration call either -- exactly one call
    # total proves no correction-retry happened for a streamed turn (unlike
    # run_turn, which would make at least one more call here).
    assert provider.call_count == 1
    # The streamed text stands as-is -- not silently swapped for something else.
    assert done["text"] == long_reply
