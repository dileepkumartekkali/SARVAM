"""LLM adapter contract + fallback router — interface only, no implementations.

Design intent (why it looks like this — read before adding a provider):

- **Streaming-first.** `stream()` is the primitive, `complete()` is a convenience
  that drains it. Barge-in (speech_to_speech §3) must cancel an in-flight LLM
  generation mid-token; an async generator gives us that for free — closing the
  generator (or cancelling the task awaiting it) stops the provider stream. A
  non-streaming `complete()`-only interface could not support barge-in.
- **One `retriable` error type**, not per-provider exceptions. The router's
  fallback logic branches on `LLMProviderError.retriable` alone; providers
  translate their own SDK errors into this at the adapter boundary so nothing
  downstream imports a vendor SDK's exception classes.
- **`name` is data, not behavior** — the router orders providers by name and
  logs which one served a turn (turn_trace), so it's part of the contract.

To add a provider later: implement this Protocol in
`llm_adapter/providers/<name>.py`, translate the SDK's errors to
`LLMProviderError`, and register it in the router's provider order. No other
module should change.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Protocol, Sequence, runtime_checkable

from ..observability.metrics import errors_total, llm_latency_seconds, llm_ttfb_seconds
from ..observability.tracing import start_span

# A message is left as a plain dict at this layer ({"role", "content"}) so no
# provider SDK's message type leaks into the interface. Tighten to a pydantic
# model in Phase 2 if the shape needs validation.
#
# For tool-calling turns (see complete_with_tools below), two additional
# shapes are generic across providers — each provider's adapter translates
# these into its own native wire format:
#   assistant requesting tool calls: {"role": "assistant", "content": <str|None>,
#                                      "tool_calls": [{"id":..., "name":..., "args": {...}}]}
#   a tool's result:                 {"role": "tool", "tool_call_id": ...,
#                                      "name": ..., "content": "<result text>"}
Message = dict[str, object]


@dataclass
class ToolDefinition:
    """Provider-agnostic tool schema — real JSON Schema, not a human-readable
    string. Each provider's `complete_with_tools()` translates this into its
    own native function-calling format (OpenAI `tools`, Anthropic
    `input_schema`, Gemini `function_declarations`)."""

    name: str
    description: str
    parameters: dict  # JSON Schema "properties" object
    required: list[str] = field(default_factory=list)


@dataclass
class ToolCall:
    """One tool the model asked to call, in our generic shape — `id` must be
    echoed back on the tool-result message so the provider can correlate it
    (OpenAI/Anthropic both require this; a call without an id can't be
    answered correctly in a multi-tool-call turn)."""

    id: str
    name: str
    args: dict


@dataclass
class CompletionResult:
    """Return shape for `complete_with_tools()` — `text` may be empty if the
    model only requested tool calls with no accompanying text."""

    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)


class LLMProviderError(Exception):
    """Raised by an adapter when a provider call fails.

    `retriable=True` means the router may fall through to the next provider
    (timeout, 429, 5xx, dropped stream). `retriable=False` means a fault the
    caller must see (auth failure, malformed request) — do not mask it by
    failing over.
    """

    def __init__(self, message: str, *, retriable: bool, provider: str | None = None):
        super().__init__(message)
        self.retriable = retriable
        self.provider = provider


@runtime_checkable
class LLMProvider(Protocol):
    """One LLM backend (Grok, GPT, Claude, Gemini, Sarvam LLM).

    Adapters are async and stateless per call; any client/session pooling is an
    implementation detail behind this interface.
    """

    name: str

    def stream(
        self,
        messages: Sequence[Message],
        *,
        system: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> AsyncIterator[str]:
        """Yield response text chunks as the provider generates them.

        Must be cancellation-safe: if the consumer stops iterating (barge-in),
        the adapter must close the underlying provider stream without leaking
        the connection. Raises `LLMProviderError` on failure.
        """
        ...

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        system: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """Non-streaming convenience: drain `stream()` into one string."""
        ...

    async def complete_with_tools(
        self,
        messages: Sequence[Message],
        *,
        system: str | None = None,
        tools: Sequence[ToolDefinition] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> CompletionResult:
        """Native provider function-calling — a real API-level tool schema
        the model chooses from, not the prompted `TOOL_CALL: {...}` text
        convention. Returns structured `ToolCall`s the caller dispatches
        directly; no text parsing involved. A provider with no meaningful
        native tool support may just ignore `tools` and return
        `CompletionResult(text=..., tool_calls=[])` — the caller
        (task_agent) falls back to the text convention when no native tool
        calls come back, so this is never a hard requirement to implement
        correctly for a provider to remain usable.
        """
        ...


class LLMRouter:
    """Ordered provider list with fallback — see failure matrix in the S2S plan.

    Holds providers in preference order (first = primary). Walks that order,
    advancing to the next provider only on `LLMProviderError(retriable=True)`;
    a non-retriable error is surfaced immediately, no fallback.
    """

    def __init__(self, providers: Sequence[LLMProvider]):
        self._providers = list(providers)

    async def complete_with_fallback(
        self,
        messages: Sequence[Message],
        *,
        system: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        last_error: LLMProviderError | None = None
        start = time.monotonic()
        with start_span("llm_router", "llm.complete"):
            for provider in self._providers:
                try:
                    result = await provider.complete(
                        messages, system=system, max_tokens=max_tokens, temperature=temperature
                    )
                    llm_latency_seconds.observe(time.monotonic() - start)
                    return result
                except LLMProviderError as e:
                    if not e.retriable:
                        errors_total.labels(stage="llm").inc()
                        raise
                    last_error = e
                    continue
            errors_total.labels(stage="llm").inc()
            assert last_error is not None, "LLMRouter has no providers configured"
            raise last_error

    async def complete_with_tools_and_fallback(
        self,
        messages: Sequence[Message],
        *,
        system: str | None = None,
        tools: Sequence[ToolDefinition] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> CompletionResult:
        """Same fallback semantics as `complete_with_fallback`, for the
        native tool-calling path task_agent's reasoning loop uses."""
        last_error: LLMProviderError | None = None
        start = time.monotonic()
        with start_span("llm_router", "llm.complete_with_tools"):
            for provider in self._providers:
                try:
                    result = await provider.complete_with_tools(
                        messages, system=system, tools=tools, max_tokens=max_tokens, temperature=temperature
                    )
                    llm_latency_seconds.observe(time.monotonic() - start)
                    return result
                except LLMProviderError as e:
                    if not e.retriable:
                        errors_total.labels(stage="llm").inc()
                        raise
                    last_error = e
                    continue
            errors_total.labels(stage="llm").inc()
            assert last_error is not None, "LLMRouter has no providers configured"
            raise last_error

    async def stream_with_fallback(
        self,
        messages: Sequence[Message],
        *,
        system: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> AsyncIterator[str]:
        """Stream with fallback, bounded by what's already been spoken.

        Fallback only applies before the first chunk is yielded — once tokens
        have gone out (and are potentially already being sent to TTS), we
        cannot retroactively switch providers without garbling or duplicating
        output. A failure after the first chunk always raises, even if
        retriable.
        """
        last_error: LLMProviderError | None = None
        start = time.monotonic()
        with start_span("llm_router", "llm.stream"):
            for provider in self._providers:
                started = False
                try:
                    async for chunk in provider.stream(
                        messages, system=system, max_tokens=max_tokens, temperature=temperature
                    ):
                        if not started:
                            llm_ttfb_seconds.observe(time.monotonic() - start)
                        started = True
                        yield chunk
                    return
                except LLMProviderError as e:
                    if started or not e.retriable:
                        errors_total.labels(stage="llm").inc()
                        raise
                    last_error = e
                    continue
            errors_total.labels(stage="llm").inc()
            assert last_error is not None, "LLMRouter has no providers configured"
            raise last_error
