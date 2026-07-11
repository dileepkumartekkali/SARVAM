"""Anthropic (Claude) adapter — Messages API.

Distinct from the OpenAI-compatible providers: auth is an `x-api-key` header
plus `anthropic-version` (not `Authorization: Bearer`), and streaming events
are `content_block_delta` objects (`delta.text`), not `choices[].delta`.

`complete_with_tools()`'s message/tool translation is written against
Anthropic's documented Messages API tool-use format (`tool_use`/
`tool_result` content blocks) — **not verified against a live Anthropic
account** in this session (no ANTHROPIC_API_KEY configured here; only the
Grok/OpenAI-compatible provider has been exercised against a real account).
Confirm against a live key before relying on this path in production.
"""

from __future__ import annotations

import json
import os
from typing import AsyncIterator, Sequence

import httpx

from .._http_common import status_retriable
from .._sse import iter_sse_data
from ..base import CompletionResult, LLMProviderError, Message, ToolCall, ToolDefinition


class AnthropicProvider:
    def __init__(
        self,
        *,
        model: str = "claude-sonnet-5",
        api_key_env: str = "ANTHROPIC_API_KEY",
        base_url: str = "https://api.anthropic.com",
        timeout: float = 30.0,
        transport: httpx.BaseTransport | None = None,
    ):
        self.name = "claude"
        self._model = model
        self._api_key_env = api_key_env
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._transport = transport

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
        body: dict[str, object] = {
            "model": self._model,
            "messages": list(messages),
            "max_tokens": max_tokens or 1024,
            "stream": True,
        }
        if system is not None:
            body["system"] = system
        if temperature is not None:
            body["temperature"] = temperature
        headers = {"x-api-key": self._api_key(), "anthropic-version": "2023-06-01"}

        try:
            async with httpx.AsyncClient(timeout=self._timeout, transport=self._transport) as client:
                async with client.stream(
                    "POST", f"{self._base_url}/v1/messages", json=body, headers=headers
                ) as resp:
                    if resp.status_code != 200:
                        await resp.aread()
                        raise LLMProviderError(
                            f"{self.name} returned {resp.status_code}: {resp.text}",
                            retriable=status_retriable(resp.status_code),
                            provider=self.name,
                        )
                    async for data in iter_sse_data(resp):
                        event = json.loads(data)
                        if event.get("type") == "content_block_delta":
                            text = event.get("delta", {}).get("text")
                            if text:
                                yield text
        except httpx.TimeoutException as e:
            raise LLMProviderError(f"{self.name} timeout: {e}", retriable=True, provider=self.name) from e
        except httpx.TransportError as e:
            raise LLMProviderError(
                f"{self.name} transport error: {e}", retriable=True, provider=self.name
            ) from e

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
        """Generic tool-call/tool-result shapes -> Claude's `tool_use`/
        `tool_result` content-block format. Tool results are role "user" for
        Claude, not "tool" — a real, documented difference from OpenAI's
        convention, not an oversight."""
        if message.get("role") == "assistant" and message.get("tool_calls"):
            content = []
            if message.get("content"):
                content.append({"type": "text", "text": message["content"]})
            for tc in message["tool_calls"]:
                content.append({"type": "tool_use", "id": tc["id"], "name": tc["name"], "input": tc["args"]})
            return {"role": "assistant", "content": content}
        if message.get("role") == "tool":
            return {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": message["tool_call_id"], "content": message["content"]}
                ],
            }
        return dict(message)

    @staticmethod
    def _tool_definitions_to_wire(tools: Sequence[ToolDefinition]) -> list[dict]:
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": {"type": "object", "properties": t.parameters, "required": t.required},
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
        body: dict[str, object] = {
            "model": self._model,
            "messages": [self._to_wire_message(m) for m in messages],
            "max_tokens": max_tokens or 1024,
            "stream": False,
        }
        if system is not None:
            body["system"] = system
        if temperature is not None:
            body["temperature"] = temperature
        if tools:
            body["tools"] = self._tool_definitions_to_wire(tools)
        headers = {"x-api-key": self._api_key(), "anthropic-version": "2023-06-01"}

        try:
            async with httpx.AsyncClient(timeout=self._timeout, transport=self._transport) as client:
                resp = await client.post(f"{self._base_url}/v1/messages", json=body, headers=headers)
                if resp.status_code != 200:
                    raise LLMProviderError(
                        f"{self.name} returned {resp.status_code}: {resp.text}",
                        retriable=status_retriable(resp.status_code),
                        provider=self.name,
                    )
                content_blocks = resp.json()["content"]
        except httpx.TimeoutException as e:
            raise LLMProviderError(f"{self.name} timeout: {e}", retriable=True, provider=self.name) from e
        except httpx.TransportError as e:
            raise LLMProviderError(f"{self.name} transport error: {e}", retriable=True, provider=self.name) from e

        text = "".join(b["text"] for b in content_blocks if b.get("type") == "text")
        tool_calls = [
            ToolCall(id=b["id"], name=b["name"], args=b["input"])
            for b in content_blocks
            if b.get("type") == "tool_use"
        ]
        return CompletionResult(text=text, tool_calls=tool_calls)
