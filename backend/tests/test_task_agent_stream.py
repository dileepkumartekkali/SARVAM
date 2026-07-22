"""stream_turn — the streaming counterpart to run_turn. Covers: plain-reply
happy path (chunks stream through, self-check's reviewer call still runs),
falling back to run_turn on a sniffed TOOL_CALL: prefix (nothing streamed
was ever wrong since nothing was yielded before the fallback), and a
self-check failure being logged rather than corrected (no regeneration call
— the accepted trade-off vs. run_turn's correction retry)."""

from agent_core.agents.task_agent import stream_turn
from agent_core.llm_adapter import LLMRouter
from agent_core.llm_adapter.base import CompletionResult, LLMProviderError
from agent_core.supervisor.state import Mode, SessionState

from ._fakes import FakeProvider, ScriptedProvider


def _session(mode=Mode.TEXT_TO_TEXT) -> SessionState:
    return SessionState(session_id="s", conversation_id="c", thread_id="t", mode=mode)


async def noop_tool(**kwargs) -> str:
    return "ok"


class _ToolCallThenProviderFailure:
    """Sniffed as a tool call by stream_turn (its `stream()`), but run_turn's
    OWN subsequent dispatch call (`complete_with_tools`) fails entirely --
    reproduces the real bug: run_turn returns TurnResult(error=True) from
    deep inside its tool loop, which stream_turn's fallback branch must not
    treat as a real answer."""

    name = "fails-after-tool-call"

    async def stream(self, messages, *, system=None, max_tokens=None, temperature=None):
        yield 'TOOL_CALL: {"name": "noop", "args": {}}'

    async def complete_with_tools(self, messages, *, system=None, tools=None, max_tokens=None, temperature=None):
        raise LLMProviderError("boom", retriable=False)

    async def complete(self, messages, *, system=None, max_tokens=None, temperature=None):
        raise LLMProviderError("boom", retriable=False)


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


async def test_stream_turn_catches_tool_call_even_with_a_conversational_lead_in():
    """Real bug hit live: the model doesn't always emit TOOL_CALL: as the
    ENTIRE reply -- it was observed prefixing "I'll search for that." /
    "Let me check." first. A `startswith`-only sniff never catches that, so
    the lead-in AND the raw TOOL_CALL JSON leaked to the UI and got spoken
    by TTS. Same four-call shape as the prefix-only case above."""
    provider = ScriptedProvider(
        [
            'I\'ll search for that.TOOL_CALL: {"name": "noop", "args": {}}',
            'TOOL_CALL: {"name": "noop", "args": {}}',
            "done",
            "OK",
        ]
    )
    router = LLMRouter([provider])

    events = [
        e async for e in stream_turn(_session(), router, "what time is it", tools={"noop": noop_tool})
    ]

    for e in events:
        if e["type"] == "text_delta":
            assert "TOOL_CALL" not in e["text"]
            assert "search for that" not in e["text"]

    done = events[-1]
    assert done["type"] == "done"
    assert done["text"] == "done"
    assert done["tool_call_count"] == 1


class _ToolCallSplitAcrossNewline:
    """Simulates real token-by-token streaming where a newline arrives in
    an EARLIER delta than the TOOL_CALL marker that follows it on its own
    line -- exactly the shape that leaked in production even after the
    startswith-vs-in fix (ScriptedProvider yields a whole reply as one
    atomic chunk, which can't reproduce this; a real provider's stream
    can't be relied on to deliver "line 1\nline 2" as a single delta)."""

    name = "split-across-newline"

    def __init__(self):
        self._dispatch_call_count = 0

    async def stream(self, messages, *, system=None, max_tokens=None, temperature=None):
        yield "Let me look that up.\n"  # delta 1 -- ends in a newline, no marker yet
        yield 'TOOL_CALL: {"name": "noop", "args": {}}'  # delta 2 -- arrives separately, after it

    async def complete_with_tools(self, messages, *, system=None, tools=None, max_tokens=None, temperature=None):
        self._dispatch_call_count += 1
        text = 'TOOL_CALL: {"name": "noop", "args": {}}' if self._dispatch_call_count == 1 else "done"
        return CompletionResult(text=text, tool_calls=[])

    async def complete(self, messages, *, system=None, max_tokens=None, temperature=None):
        return "OK"  # self-check's reviewer call


async def test_stream_turn_catches_tool_call_on_its_own_line_after_a_lead_in():
    """Real bug hit live, a SECOND time: the previous fix (check `in` not
    `startswith`) still leaked in production for a reply shaped like
    "<lead-in line>\nTOOL_CALL: {...}" -- the newline after the lead-in
    used to trigger the sniff's decision on THAT LINE ALONE (which has no
    marker), permanently stopping all further checking before the actual
    TOOL_CALL line ever arrived, delivered as a SEPARATE delta. A newline
    can't be treated as a "safe, decided" signal when the model
    legitimately puts its explanation and its tool call on separate lines."""
    router = LLMRouter([_ToolCallSplitAcrossNewline()])

    events = [
        e async for e in stream_turn(_session(), router, "what time is it", tools={"noop": noop_tool})
    ]

    for e in events:
        if e["type"] == "text_delta":
            assert "TOOL_CALL" not in e["text"]
            assert "look that up" not in e["text"]

    done = events[-1]
    assert done["type"] == "done"
    assert done["text"] == "done"
    assert done["tool_call_count"] == 1


async def test_stream_turn_never_leaks_run_turns_internal_apology_as_a_delta():
    """Real bug hit live: run_turn's OWN tool-dispatch loop can hit
    LLMProviderError and return the apology as TurnResult.error=True --
    stream_turn's tool-call fallback branch used to yield that unconditionally
    as a text_delta, so the fake apology got shown/spoken exactly like a
    real answer. Must now suppress the delta and mark the done event as an
    error, same contract as the top-level failure path."""
    router = LLMRouter([_ToolCallThenProviderFailure()])

    events = [
        e async for e in stream_turn(_session(), router, "what time is it", tools={"noop": noop_tool})
    ]

    assert all(e["type"] != "text_delta" for e in events)
    done = events[-1]
    assert done["type"] == "done"
    assert done["error"] is True
    assert "trouble" in done["text"].lower()


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


async def test_stream_turn_provider_failure_never_yields_a_text_delta():
    """The apology on total provider failure is not a real answer -- it must
    never be spoken aloud (voice mode's TTS socket is fed from text_delta
    events) or persisted as chat history (main.py skips both when the final
    "done" event's error flag is set). It only ever reaches the caller via
    the "done" event's own text field."""
    provider = FakeProvider("x", error=LLMProviderError("boom", retriable=False))
    router = LLMRouter([provider])

    events = [e async for e in stream_turn(_session(), router, "hi")]

    assert all(e["type"] != "text_delta" for e in events)
    done = events[-1]
    assert done["type"] == "done"
    assert done["error"] is True
    assert "trouble" in done["text"].lower()
