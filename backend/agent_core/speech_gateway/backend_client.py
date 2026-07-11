"""HTTP bridge from the Speech Gateway to the main backend's `/chat`.

Two processes, two key sets (S2S plan §6: gateway holds Sarvam/Azure keys,
backend holds LLM keys) — the gateway never talks to an LLM directly. It
relays a finished transcript to the backend exactly like a text client would,
over one shared HTTP connection pool, then speaks back whatever text comes
back. The user's own bearer token is forwarded unchanged, so auth/RBAC is
enforced once, by the backend, never duplicated here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import httpx

BACKEND_CHAT_URL = os.environ.get("BACKEND_CHAT_URL", "http://localhost:9000/chat")
_TIMEOUT_SECONDS = float(os.environ.get("BACKEND_CHAT_TIMEOUT_SECONDS", "20"))

# One pooled client for the process lifetime — constructing it never touches
# the network, same invariant as the Sarvam clients constructed at import
# time elsewhere in this service.
_http_client = httpx.AsyncClient(timeout=_TIMEOUT_SECONDS)


@dataclass
class BackendChatReply:
    text: str
    pending_confirmation_token: str | None = None
    pending_confirmation_tool: str | None = None


class BackendChatError(Exception):
    """The backend call failed or timed out. Callers speak a fixed apology
    (failure_policy.LLM_TIMEOUT_APOLOGY) rather than propagate this to the
    user as-is — never dead air."""


async def call_backend_chat(
    *,
    message: str,
    session_id: str,
    conversation_id: str,
    thread_id: str,
    language: str | None,
    auth_token: str,
    confirmation_token: str | None = None,
) -> BackendChatReply:
    """Relays one finished utterance to `/chat` as `mode="speech_to_speech"`,
    so the backend's confirmation gate (Mode.is_voice) actually applies to
    write-scope tools reached over voice. `confirmation_token` is simply
    whatever the PREVIOUS turn's `pending_confirmation.token` was (or None) —
    the backend decides whether this turn's tool call matches it; the gateway
    never inspects or interprets the token itself.
    """
    payload = {
        "session_id": session_id,
        "conversation_id": conversation_id,
        "thread_id": thread_id,
        "message": message,
        "mode": "speech_to_speech",
        "stt_language_hint": language,
        "confirmation_token": confirmation_token,
    }
    try:
        response = await _http_client.post(
            BACKEND_CHAT_URL,
            json=payload,
            headers={"Authorization": f"Bearer {auth_token}"},
        )
        response.raise_for_status()
    except httpx.HTTPError as e:
        raise BackendChatError(str(e)) from e

    body = response.json()
    pending = body.get("pending_confirmation")
    return BackendChatReply(
        text=body["response"],
        pending_confirmation_token=pending["token"] if pending else None,
        pending_confirmation_tool=pending["tool_name"] if pending else None,
    )
