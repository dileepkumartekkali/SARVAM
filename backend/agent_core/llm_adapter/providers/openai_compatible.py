"""Adapter for any OpenAI-`/chat/completions`-compatible endpoint.

Grok (x.ai), OpenAI GPT, and Sarvam's LLM API all speak the same request/
response schema (`messages`, `stream=true`, `choices[0].delta.content`) — one
implementation parameterized by base URL / model / API-key env var, instead of
three near-identical adapters that would drift out of sync.

`complete_with_tools()` is a genuinely separate, non-streaming request
(`stream: false`) rather than draining `stream()` — OpenAI's tool_calls come
back as one structured field on the final message, which is far simpler to
parse from a single JSON response than to reconstruct from streamed deltas.
This is the one provider in this file verified against a live account (Grok)
— see tests/test_agentic_loop_real_tools.py and the session's own manual
verification (calculate(17*23) -> 391, executed for real).
"""

from __future__ import annotations

import json
import os
from typing import AsyncIterator, Sequence

import httpx

from .._http_common import status_retriable
from .._sse import iter_sse_data
from ..base import CompletionResult, LLMProviderError, Message, ToolCall, ToolDefinition


class OpenAICompatibleProvider:
    def __init__(
        self,
        *,
        name: str,
        base_url: str,
        model: str,
        api_key_env: str,
        timeout: float = 30.0,
        transport: httpx.BaseTransport | None = None,
        extra_body: dict[str, object] | None = None,
    ):
        self.name = name
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key_env = api_key_env
        self._timeout = timeout
        self._transport = transport  # test-injection hook only; None in prod
        # Provider-specific request fields with no generic equivalent — e.g.
        # Sarvam's `reasoning_effort` (its models are chain-of-thought
        # reasoning models that spend hundreds of tokens on an internal
        # `reasoning_content` field before any real answer; this bounds that).
        self._extra_body = extra_body or {}

    def _api_key(self) -> str:
        key = os.environ.get(self._api_key_env)
        if not key:
            raise LLMProviderError(
                f"{self._api_key_env} not set", retriable=False, provider=self.name
            )
        return key

    async def stream(
        self,
        messages: Sequence[Message],
        *,
        system: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> AsyncIterator[str]:
        payload_messages = ([{"role": "system", "content": system}] if system else []) + list(
            messages
        )
        body: dict[str, object] = {
            "model": self._model,
            "messages": payload_messages,
            "stream": True,
            **self._extra_body,
        }
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if temperature is not None:
            body["temperature"] = temperature
        headers = {"Authorization": f"Bearer {self._api_key()}"}

        yielded_any = False
        truncated = False
        try:
            async with httpx.AsyncClient(timeout=self._timeout, transport=self._transport) as client:
                async with client.stream(
                    "POST", f"{self._base_url}/chat/completions", json=body, headers=headers
                ) as resp:
                    if resp.status_code != 200:
                        await resp.aread()
                        raise LLMProviderError(
                            f"{self.name} returned {resp.status_code}: {resp.text}",
                            retriable=status_retriable(resp.status_code),
                            provider=self.name,
                        )
                    async for data in iter_sse_data(resp):
                        chunk = json.loads(data)
                        choices = chunk.get("choices") or []
                        if not choices:
                            # A real shape hit live against Sarvam: a trailing
                            # chunk (usage stats/keep-alive) with an EMPTY
                            # choices array — `choices[0]` on it is a crash
                            # a real API can actually send, not hypothetical.
                            continue
                        choice = choices[0]
                        # A reasoning model (e.g. Sarvam's) puts chain-of-
                        # thought in a separate `reasoning_content` field,
                        # never `content` — deliberately ignored here so it
                        # never leaks into the actual answer.
                        text = choice.get("delta", {}).get("content")
                        if text:
                            yielded_any = True
                            yield text
                        truncated = choice.get("finish_reason") == "length"
        except httpx.TimeoutException as e:
            raise LLMProviderError(f"{self.name} timeout: {e}", retriable=True, provider=self.name) from e
        except httpx.TransportError as e:
            raise LLMProviderError(
                f"{self.name} transport error: {e}", retriable=True, provider=self.name
            ) from e

        if truncated and not yielded_any:
            # Hit max_tokens while still inside `reasoning_content` and never
            # reached a real answer — a 200 OK, but not a usable one. Treated
            # as a retriable failure so the router's existing fallback chain
            # (already built for exactly this) engages, rather than silently
            # handing the caller an empty string as if it were a real reply.
            raise LLMProviderError(
                f"{self.name} truncated at max_tokens before producing any content "
                "(the model was still reasoning)",
                retriable=True,
                provider=self.name,
            )

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        system: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        chunks = [
            c
            async for c in self.stream(
                messages, system=system, max_tokens=max_tokens, temperature=temperature
            )
        ]
        return "".join(chunks)

    @staticmethod
    def _to_wire_message(message: Message) -> dict:
        """Translates this codebase's generic tool-call/tool-result message
        shapes (see llm_adapter/base.py's Message docstring) into OpenAI's
        native wire format — a no-op for plain user/assistant text messages."""
        if message.get("role") == "assistant" and message.get("tool_calls"):
            return {
                "role": "assistant",
                "content": message.get("content"),
                "tool_calls": [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {"name": tc["name"], "arguments": json.dumps(tc["args"])},
                    }
                    for tc in message["tool_calls"]
                ],
            }
        if message.get("role") == "tool":
            return {"role": "tool", "tool_call_id": message["tool_call_id"], "content": message["content"]}
        return dict(message)

    @staticmethod
    def _tool_definitions_to_wire(tools: Sequence[ToolDefinition]) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": {"type": "object", "properties": t.parameters, "required": t.required},
                },
            }
            for t in tools
        ]

    async def complete_with_tools(
        self,
        messages: Sequence[Message],
        *,
        system: str | None = None,
        tools: Sequence[ToolDefinition] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> CompletionResult:
        payload_messages = ([{"role": "system", "content": system}] if system else []) + [
            self._to_wire_message(m) for m in messages
        ]
        body: dict[str, object] = {
            "model": self._model,
            "messages": payload_messages,
            "stream": False,
            **self._extra_body,
        }
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if temperature is not None:
            body["temperature"] = temperature
        if tools:
            body["tools"] = self._tool_definitions_to_wire(tools)
        headers = {"Authorization": f"Bearer {self._api_key()}"}

        try:
            async with httpx.AsyncClient(timeout=self._timeout, transport=self._transport) as client:
                resp = await client.post(f"{self._base_url}/chat/completions", json=body, headers=headers)
                if resp.status_code != 200:
                    raise LLMProviderError(
                        f"{self.name} returned {resp.status_code}: {resp.text}",
                        retriable=status_retriable(resp.status_code),
                        provider=self.name,
                    )
                choices = resp.json().get("choices") or []
                if not choices:
                    # Same empty-choices shape stream() above already guards
                    # against (a real one hit live against Sarvam) -- this
                    # non-streaming path had no equivalent guard, so it would
                    # crash with a raw IndexError instead of a retriable
                    # LLMProviderError, escaping the router's fallback chain.
                    raise LLMProviderError(
                        f"{self.name} returned a 200 with no choices", retriable=True, provider=self.name
                    )
                choice = choices[0]
                message = choice["message"]
        except httpx.TimeoutException as e:
            raise LLMProviderError(f"{self.name} timeout: {e}", retriable=True, provider=self.name) from e
        except httpx.TransportError as e:
            raise LLMProviderError(f"{self.name} transport error: {e}", retriable=True, provider=self.name) from e

        # A reasoning model (e.g. Sarvam's) puts chain-of-thought in a separate
        # `reasoning_content` field, never `content` or `tool_calls` — if it hit
        # max_tokens while still reasoning, both come back empty on an
        # otherwise-200-OK response. That's not a usable answer; treated as a
        # retriable failure so the router's fallback chain engages instead of
        # handing the caller an empty string as if it were real.
        no_content = not message.get("content") and not message.get("tool_calls")
        if choice.get("finish_reason") == "length" and no_content:
            raise LLMProviderError(
                f"{self.name} truncated at max_tokens before producing any content "
                "(the model was still reasoning)",
                retriable=True,
                provider=self.name,
            )

        tool_calls = []
        for tc in message.get("tool_calls") or []:
            try:
                # A real gap: malformed/empty tool-call arguments (a
                # provider returning "" for a zero-argument call, or
                # otherwise-invalid JSON) raised a bare json.JSONDecodeError
                # here -- not an LLMProviderError, so nothing downstream
                # (the router's fallback loop, task_agent's exception
                # handling) ever caught it. It escaped as an uncaught 500
                # with no fallback attempted, defeating the one error type
                # this whole adapter layer exists to normalize to (see
                # base.py's own module docstring).
                args = json.loads(tc["function"]["arguments"] or "{}")
            except json.JSONDecodeError as e:
                raise LLMProviderError(
                    f"{self.name} returned malformed tool-call arguments: {e}", retriable=False, provider=self.name
                ) from e
            tool_calls.append(ToolCall(id=tc["id"], name=tc["function"]["name"], args=args))
        return CompletionResult(text=message.get("content") or "", tool_calls=tool_calls)
