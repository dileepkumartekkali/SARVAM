"""Google Gemini adapter — `streamGenerateContent` over SSE.

Distinct from the OpenAI-compatible providers: request body uses
`contents[].parts[].text` (not `messages`), auth is a `key` query param (not a
header), and system prompt is a top-level `system_instruction`.

`complete_with_tools()`'s message/tool translation is written against
Gemini's documented `function_declarations`/`functionCall`/`functionResponse`
format. Auth and endpoint shape verified live against a real key; actual
model access was NOT — every model tried (gemini-2.5-flash, gemini-2.5-flash-
lite, gemini-2.0-flash, gemini-2.0-flash-lite) returned either 404 ("no longer
available to new users") or 429 with `limit: 0` (zero free-tier quota) — an
account/billing issue on the Google Cloud project behind the key, not
something fixable here. `complete_with_tools()`'s message/tool-call parsing
itself remains unverified against a real 200 response.
"""

from __future__ import annotations

import json
import os
from typing import AsyncIterator, Sequence

import httpx

from .._http_common import status_retriable
from .._sse import iter_sse_data
from ..base import CompletionResult, LLMProviderError, Message, ToolCall, ToolDefinition


def _gemini_retriable(status_code: int) -> bool:
    """404 is generally a non-retriable caller bug (`_http_common.py`) — but
    for Gemini specifically, a live 404 means "this model is no longer
    available to new users," a provider-availability fact just like a 429 or
    5xx, not a malformed request. Verified live: `gemini-2.5-flash` (the
    configured default) returns exactly this 404 with a zero-quota Gemini
    account — treating it as non-retriable would hard-crash every single
    turn instead of falling back to the next configured provider.
    """
    return status_retriable(status_code) or status_code == 404


class GeminiProvider:
    def __init__(
        self,
        *,
        model: str = "gemini-2.5-flash",
        api_key_env: str = "GEMINI_API_KEY",
        base_url: str = "https://generativelanguage.googleapis.com",
        timeout: float = 30.0,
        transport: httpx.BaseTransport | None = None,
    ):
        self.name = "gemini"
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

    @staticmethod
    def _to_contents(messages: Sequence[Message]) -> list[dict]:
        # Gemini has no "assistant" role — it's "model".
        return [
            {"role": "model" if m.get("role") == "assistant" else "user", "parts": [{"text": m["content"]}]}
            for m in messages
        ]

    async def stream(
        self,
        messages: Sequence[Message],
        *,
        system: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> AsyncIterator[str]:
        body: dict[str, object] = {"contents": self._to_contents(messages)}
        if system is not None:
            body["system_instruction"] = {"parts": [{"text": system}]}
        gen_config: dict[str, object] = {}
        if max_tokens is not None:
            gen_config["maxOutputTokens"] = max_tokens
        if temperature is not None:
            gen_config["temperature"] = temperature
        if gen_config:
            body["generationConfig"] = gen_config

        url = f"{self._base_url}/v1beta/models/{self._model}:streamGenerateContent"
        params = {"key": self._api_key(), "alt": "sse"}

        try:
            async with httpx.AsyncClient(timeout=self._timeout, transport=self._transport) as client:
                async with client.stream("POST", url, json=body, params=params) as resp:
                    if resp.status_code != 200:
                        await resp.aread()
                        raise LLMProviderError(
                            f"{self.name} returned {resp.status_code}: {resp.text}",
                            retriable=_gemini_retriable(resp.status_code),
                            provider=self.name,
                        )
                    async for data in iter_sse_data(resp):
                        chunk = json.loads(data)
                        for candidate in chunk.get("candidates", []):
                            for part in candidate.get("content", {}).get("parts", []):
                                text = part.get("text")
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
    def _to_wire_content(message: Message) -> dict:
        """Generic tool-call/tool-result shapes -> Gemini's `functionCall`/
        `functionResponse` parts. Gemini's assistant role is "model"; tool
        results use role "function", not "tool"."""
        if message.get("role") == "assistant" and message.get("tool_calls"):
            parts = []
            if message.get("content"):
                parts.append({"text": message["content"]})
            for tc in message["tool_calls"]:
                parts.append({"functionCall": {"name": tc["name"], "args": tc["args"]}})
            return {"role": "model", "parts": parts}
        if message.get("role") == "tool":
            return {
                "role": "function",
                "parts": [{"functionResponse": {"name": message["name"], "response": {"result": message["content"]}}}],
            }
        role = "model" if message.get("role") == "assistant" else "user"
        return {"role": role, "parts": [{"text": message["content"]}]}

    @staticmethod
    def _tool_definitions_to_wire(tools: Sequence[ToolDefinition]) -> list[dict]:
        return [
            {
                "function_declarations": [
                    {
                        "name": t.name,
                        "description": t.description,
                        "parameters": {"type": "object", "properties": t.parameters, "required": t.required},
                    }
                    for t in tools
                ]
            }
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
        body: dict[str, object] = {"contents": [self._to_wire_content(m) for m in messages]}
        if system is not None:
            body["system_instruction"] = {"parts": [{"text": system}]}
        gen_config: dict[str, object] = {}
        if max_tokens is not None:
            gen_config["maxOutputTokens"] = max_tokens
        if temperature is not None:
            gen_config["temperature"] = temperature
        if gen_config:
            body["generationConfig"] = gen_config
        if tools:
            body["tools"] = self._tool_definitions_to_wire(tools)

        url = f"{self._base_url}/v1beta/models/{self._model}:generateContent"
        params = {"key": self._api_key()}

        try:
            async with httpx.AsyncClient(timeout=self._timeout, transport=self._transport) as client:
                resp = await client.post(url, json=body, params=params)
                if resp.status_code != 200:
                    raise LLMProviderError(
                        f"{self.name} returned {resp.status_code}: {resp.text}",
                        retriable=_gemini_retriable(resp.status_code),
                        provider=self.name,
                    )
                parts = resp.json()["candidates"][0]["content"]["parts"]
        except httpx.TimeoutException as e:
            raise LLMProviderError(f"{self.name} timeout: {e}", retriable=True, provider=self.name) from e
        except httpx.TransportError as e:
            raise LLMProviderError(f"{self.name} transport error: {e}", retriable=True, provider=self.name) from e

        text = "".join(p["text"] for p in parts if "text" in p)
        tool_calls = [
            ToolCall(id=p["functionCall"]["name"], name=p["functionCall"]["name"], args=p["functionCall"]["args"])
            for p in parts
            if "functionCall" in p
        ]
        return CompletionResult(text=text, tool_calls=tool_calls)
