"""FastAPI app entrypoint.

Text→Text chat via LangGraph, real providers configured from env. Voice
WebSocket routes live in the Speech Gateway (its own service).
"""

from __future__ import annotations

import os
import time
from typing import Literal

import jwt
from fastapi import Depends, FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ..llm_adapter import build_router_from_env
from ..observability.logging_config import configure_logging
from ..observability.metrics import metrics_response
from ..observability.tracing import init_tracing
from ..security.auth import AuthConfig, Principal, get_current_principal
from ..supervisor.graph import build_text_graph
from ..supervisor.state import Mode, SessionState
from ..tools import build_default_registry

configure_logging()
init_tracing("backend")

app = FastAPI(title="MAAV / Vaani backend", version="0.1.0")

# Explicit allow-list only — no wildcard-with-credentials, which browsers
# treat as a CORS misconfiguration anyway. Empty by default (deny all
# cross-origin) until a real frontend origin is configured.
_allowed_origins = [o for o in os.environ.get("CORS_ALLOWED_ORIGINS", "").split(",") if o]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)

# Built once at import time. Constructing a router/graph never touches the
# network or requires API keys (adapters only read their key env var when a
# call is actually made) — so this can't break `docker-compose up` even with
# no provider keys configured.
if os.environ.get("MAAV_LOAD_TEST_MODE", "").lower() == "true":
    # Never true in production — see load_test_fakes.py's docstring.
    from ..llm_adapter.base import LLMRouter
    from .load_test_fakes import LoadTestFakeProvider

    _router = LLMRouter([LoadTestFakeProvider()])
else:
    _router = build_router_from_env()

# Real tools, not an empty namespace — see agent_core/tools/builtin.py.
_tool_registry = build_default_registry()
_graph = build_text_graph(
    _router,
    tools=_tool_registry.as_dispatch_dict(),
    tool_definitions=_tool_registry.as_tool_definitions(),
    tool_manifest=_tool_registry.as_prompt_manifest(),
    write_scope_tools=_tool_registry.write_scope_names(),
)


def get_graph():
    """FastAPI dependency, overridden in tests to inject a fake-provider graph."""
    return _graph


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/metrics")
async def metrics() -> Response:
    body, content_type = metrics_response()
    return Response(content=body, media_type=content_type)


class DevLoginRequest(BaseModel):
    username: str


class DevLoginResponse(BaseModel):
    access_token: str


@app.post("/auth/dev-login", response_model=DevLoginResponse)
async def dev_login(req: DevLoginRequest) -> DevLoginResponse:
    """Dev-only token issuance so the frontend's auth flow is real and
    clickable end-to-end without a live OAuth IdP (see docs/THREAT_MODEL.md —
    real IdP integration is accepted risk for v1, not solved here). 404s
    unless DEV_AUTH_ENABLED=true, so it's never reachable by default.
    """
    if os.environ.get("DEV_AUTH_ENABLED", "").lower() != "true":
        raise HTTPException(status_code=404)
    config = AuthConfig()
    token = jwt.encode(
        {"sub": req.username, "roles": ["dev_user"], "exp": int(time.time()) + 3600},
        config.secret(),
        algorithm=config.algorithm,
    )
    return DevLoginResponse(access_token=token)


class ChatRequest(BaseModel):
    session_id: str
    conversation_id: str
    thread_id: str
    message: str
    # Determines the prompt variant AND whether write-scope tools require voice
    # confirmation (Mode.is_voice — see security/confirmation.py). The Speech
    # Gateway sets this to "speech_to_speech" when relaying a spoken turn;
    # defaults to text so every existing text-only caller is unaffected.
    mode: Literal["text_to_text", "speech_to_text", "text_to_speech", "speech_to_speech"] = "text_to_text"
    # Unverified signal only — language_agent detects language independently
    # and never adopts this as the answer; see agent_core.agents.language_agent.
    stt_language_hint: str | None = None
    # Set when resubmitting after a `pending_confirmation` response, to
    # actually execute the write-scope tool it named — see
    # security/confirmation.py. Ignored for TEXT_TO_TEXT turns (the gate only
    # applies to voice-originated turns; text already has the confirmation
    # built in — the user re-typing the request).
    confirmation_token: str | None = None


class PendingConfirmationResponse(BaseModel):
    token: str
    tool_name: str
    args: dict


class ChatResponse(BaseModel):
    response: str
    prompt_version: str
    tool_call_count: int
    self_check_ok: bool
    response_language: str | None
    language_confidence: float | None
    is_code_mixed: bool
    pending_confirmation: PendingConfirmationResponse | None = None


@app.post("/chat", response_model=ChatResponse)
async def chat(
    req: ChatRequest,
    graph=Depends(get_graph),
    principal: Principal = Depends(get_current_principal),
) -> ChatResponse:
    session = SessionState(
        session_id=req.session_id,
        conversation_id=req.conversation_id,
        thread_id=req.thread_id,
        mode=Mode(req.mode),
    )
    result = await graph.ainvoke(
        {
            "session": session,
            "user_message": req.message,
            "stt_language_hint": req.stt_language_hint,
            "confirmation_token": req.confirmation_token,
        },
        config={"configurable": {"thread_id": req.thread_id}},
    )
    detected_session: SessionState = result["session"]
    pending = result.get("pending_confirmation")
    return ChatResponse(
        response=result["response"],
        prompt_version=result["prompt_version"],
        tool_call_count=result["tool_call_count"],
        self_check_ok=result["self_check_ok"],
        response_language=detected_session.response_language,
        language_confidence=detected_session.language_confidence,
        is_code_mixed=detected_session.is_code_mixed,
        pending_confirmation=(
            PendingConfirmationResponse(token=pending.token, tool_name=pending.tool_name, args=pending.args)
            if pending is not None
            else None
        ),
    )
