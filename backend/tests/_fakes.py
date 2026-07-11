"""Fake LLMProvider stand-ins shared across test modules — no network calls.

Every fake implements `complete_with_tools()` by default as a thin wrapper
around its existing `complete()` behavior, returning
`CompletionResult(text=<that string>, tool_calls=[])` — this is what keeps
every pre-existing test (written against the `TOOL_CALL: {...}` text
convention) passing unchanged: task_agent's loop tries the native path
first, sees no `tool_calls` came back, and falls back to parsing the text
exactly as before. `NativeToolCallProvider` is the one fake that returns
*real* structured `ToolCall`s, for tests proving the native path itself.
"""

from __future__ import annotations

from agent_core.llm_adapter import LLMProviderError
from agent_core.llm_adapter.base import CompletionResult, ToolCall


class FakeProvider:
    """Replays a canned outcome (chunks and/or a scripted error)."""

    def __init__(self, name, *, chunks=None, error: LLMProviderError | None = None, fail_after: int = 0):
        self.name = name
        self._chunks = chunks or []
        self._error = error
        self._fail_after = fail_after  # raise `_error` after yielding this many chunks
        self.calls = 0

    async def stream(self, messages, *, system=None, max_tokens=None, temperature=None):
        self.calls += 1
        for i, chunk in enumerate(self._chunks):
            if self._error is not None and i == self._fail_after:
                raise self._error
            yield chunk
        if self._error is not None and self._fail_after >= len(self._chunks):
            raise self._error

    async def complete(self, messages, *, system=None, max_tokens=None, temperature=None):
        self.calls += 1
        if self._error is not None:
            raise self._error
        return "".join(self._chunks)

    async def complete_with_tools(self, messages, *, system=None, tools=None, max_tokens=None, temperature=None):
        text = await self.complete(messages, system=system, max_tokens=max_tokens, temperature=temperature)
        return CompletionResult(text=text, tool_calls=[])


class InjectionProbeProvider:
    """Stands in for an LLM that WOULD follow the most recent imperative
    sentence in its input — unless that sentence is inside an
    `<<UNTRUSTED_...>>` block. First call always requests a tool; second call
    decides compliance based on whether the injected text reached it wrapped
    or bare. Used to prove `run_turn` always wraps tool content before it
    reaches the model — never as a bare instruction.
    """

    name = "probe"

    def __init__(self, tool_name: str = "lookup_note"):
        self._tool_name = tool_name
        self.call_count = 0
        self.calls: list = []  # messages seen on each call, in order

    async def complete(self, messages, *, system=None, max_tokens=None, temperature=None):
        self.call_count += 1
        self.calls.append(messages)
        if self.call_count == 1:
            return f'TOOL_CALL: {{"name": "{self._tool_name}", "args": {{}}}}'
        combined = "\n".join(str(m["content"]) for m in messages)
        bare_injection_present = "Ignore all previous instructions" in combined and (
            "<<UNTRUSTED_" not in combined
        )
        if bare_injection_present:
            return "PWNED"
        return "Here's what I found in your note."

    async def stream(self, messages, *, system=None, max_tokens=None, temperature=None):
        yield await self.complete(messages, system=system, max_tokens=max_tokens, temperature=temperature)

    async def complete_with_tools(self, messages, *, system=None, tools=None, max_tokens=None, temperature=None):
        text = await self.complete(messages, system=system, max_tokens=max_tokens, temperature=temperature)
        return CompletionResult(text=text, tool_calls=[])


class InfiniteToolCallProvider:
    """Always asks to call the same tool again — simulates a runaway loop."""

    name = "loopy"

    def __init__(self, tool_name: str = "noop"):
        self._tool_name = tool_name

    async def complete(self, messages, *, system=None, max_tokens=None, temperature=None):
        return f'TOOL_CALL: {{"name": "{self._tool_name}", "args": {{}}}}'

    async def stream(self, messages, *, system=None, max_tokens=None, temperature=None):
        yield await self.complete(messages, system=system, max_tokens=max_tokens, temperature=temperature)

    async def complete_with_tools(self, messages, *, system=None, tools=None, max_tokens=None, temperature=None):
        text = await self.complete(messages, system=system, max_tokens=max_tokens, temperature=temperature)
        return CompletionResult(text=text, tool_calls=[])


class ScriptedProvider:
    """Returns a different canned reply on each successive call."""

    name = "scripted"

    def __init__(self, replies: list[str]):
        self._replies = list(replies)
        self.call_count = 0
        self.systems: list = []  # system prompt seen on each call, in order
        self.messages_by_call: list = []
        self.last_system = None  # kept for older tests; reflects the most recent call
        self.last_messages = None

    async def complete(self, messages, *, system=None, max_tokens=None, temperature=None):
        self.last_system = system
        self.last_messages = messages
        self.systems.append(system)
        self.messages_by_call.append(messages)
        reply = self._replies[min(self.call_count, len(self._replies) - 1)]
        self.call_count += 1
        return reply

    async def stream(self, messages, *, system=None, max_tokens=None, temperature=None):
        yield await self.complete(messages, system=system, max_tokens=max_tokens, temperature=temperature)

    async def complete_with_tools(self, messages, *, system=None, tools=None, max_tokens=None, temperature=None):
        text = await self.complete(messages, system=system, max_tokens=max_tokens, temperature=temperature)
        return CompletionResult(text=text, tool_calls=[])


class NativeToolCallProvider:
    """The one fake that returns REAL structured `ToolCall`s from
    `complete_with_tools()` — proves task_agent's native-calling path (not
    the text-convention fallback) actually dispatches tools. `tool_calls_by_step`
    is a list of lists: step i's `ToolCall`s (empty list = plain text turn,
    ending the loop with `final_texts[i]`)."""

    name = "native"

    def __init__(self, tool_calls_by_step: list[list[ToolCall]], final_texts: list[str]):
        self._tool_calls_by_step = tool_calls_by_step
        self._final_texts = final_texts
        self.call_count = 0
        self.messages_by_call: list = []
        self.tools_by_call: list = []

    async def complete(self, messages, *, system=None, max_tokens=None, temperature=None):
        # Used by _self_check's plain critique call, which doesn't need
        # tools — always says the draft is compliant so these tests can
        # focus on the tool-calling path itself.
        return "OK"

    async def stream(self, messages, *, system=None, max_tokens=None, temperature=None):
        yield await self.complete(messages, system=system, max_tokens=max_tokens, temperature=temperature)

    async def complete_with_tools(self, messages, *, system=None, tools=None, max_tokens=None, temperature=None):
        self.messages_by_call.append(messages)
        self.tools_by_call.append(tools)
        step = self.call_count
        self.call_count += 1
        calls = self._tool_calls_by_step[step] if step < len(self._tool_calls_by_step) else []
        text = self._final_texts[step] if step < len(self._final_texts) else ""
        return CompletionResult(text=text, tool_calls=calls)


class PoisonProvider:
    """Raises if ever called — proves a code path never reaches task_agent's LLM."""

    name = "poison"

    def __init__(self):
        self.calls = 0

    async def complete(self, messages, *, system=None, max_tokens=None, temperature=None):
        self.calls += 1
        raise AssertionError("this provider must never be called")

    async def stream(self, messages, *, system=None, max_tokens=None, temperature=None):
        self.calls += 1
        raise AssertionError("this provider must never be called")
        yield  # pragma: no cover — unreachable, satisfies async generator shape

    async def complete_with_tools(self, messages, *, system=None, tools=None, max_tokens=None, temperature=None):
        self.calls += 1
        raise AssertionError("this provider must never be called")
