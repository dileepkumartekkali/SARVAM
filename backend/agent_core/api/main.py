"""FastAPI app entrypoint.

Text→Text chat via LangGraph, real providers configured from env. Voice
WebSocket routes live in the Speech Gateway (its own service).
"""

from __future__ import annotations

import asyncio
import os
import uuid
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ..llm_adapter import build_router_from_env
from ..observability.logging_config import configure_logging
from ..observability.metrics import metrics_response
from ..observability.tracing import init_tracing
from ..persistence import chat_store
from ..security.auth import Principal, get_current_principal
from ..supervisor.graph import build_text_graph
from ..supervisor.state import Mode, SessionState
from ..tools import build_default_registry

configure_logging()
init_tracing("backend")


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    # Pays the Postgres connection-pool setup cost once at boot instead of on
    # whichever user's request happens to be first after a cold start (Render
    # free tier idles the whole process) — a real, measured chunk of the
    # "first load after refresh is slow" latency.
    await chat_store.warm_up()
    yield


app = FastAPI(title="MAAV / Mvoice backend", version="0.1.0", lifespan=_lifespan)

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
    # The persisted assistant message's row id (null if persistence isn't
    # configured) — the frontend needs this to later attach a replay audio
    # path to the right row once TTS finishes (see chat_store.set_message_audio_path).
    message_id: str | None = None


@app.post("/chat", response_model=ChatResponse)
async def chat(
    req: ChatRequest,
    background_tasks: BackgroundTasks,
    graph=Depends(get_graph),
    principal: Principal = Depends(get_current_principal),
) -> ChatResponse:
    session = SessionState(
        session_id=req.session_id,
        conversation_id=req.conversation_id,
        thread_id=req.thread_id,
        mode=Mode(req.mode),
    )
    persistence_on = chat_store.is_configured()

    # The ownership check is a DB round-trip that has nothing to do with what
    # the LLM produces -- running it concurrently with graph.ainvoke() (not
    # before it) hides its latency entirely behind the LLM call, which is
    # always far longer. A wasted LLM call only happens in the (never
    # expected in normal use) case of a bogus/foreign conversation_id.
    owned_check = chat_store.get_conversation(req.conversation_id, principal.subject) if persistence_on else None
    if owned_check is not None:
        owned, result = await asyncio.gather(
            owned_check,
            graph.ainvoke(
                {
                    "session": session,
                    "user_message": req.message,
                    "stt_language_hint": req.stt_language_hint,
                    "confirmation_token": req.confirmation_token,
                },
                config={"configurable": {"thread_id": req.thread_id}},
            ),
        )
        if owned is None:
            raise HTTPException(status_code=404, detail="conversation not found")
    else:
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

    # Persisting both sides of the turn is durability, not something the
    # caller is waiting on -- it runs as a background task AFTER the response
    # below is already sent. The id is generated here (not by Postgres) so it
    # can be returned immediately for the audio-replay upload to target.
    assistant_message_id = None
    if persistence_on:
        assistant_message_id = str(uuid.uuid4())
        background_tasks.add_task(
            chat_store.record_turn,
            req.conversation_id,
            principal.subject,
            req.message,
            result["response"],
            detected_session.response_language,
            assistant_message_id,
        )

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
        message_id=assistant_message_id,
    )


class StoredMessage(BaseModel):
    id: str
    role: Literal["user", "assistant"]
    content: str
    audio_path: str | None
    response_language: str | None


class ConversationSummary(BaseModel):
    id: str
    title: str | None
    updated_at: str


class CreateConversationResponse(BaseModel):
    id: str


@app.get("/conversations", response_model=list[ConversationSummary])
async def list_conversations(principal: Principal = Depends(get_current_principal)) -> list[ConversationSummary]:
    """Ordered most-recently-active first — what populates the sidebar."""
    rows = await chat_store.list_conversations(principal.subject)
    return [
        ConversationSummary(id=str(r["id"]), title=r["title"], updated_at=r["updated_at"].isoformat())
        for r in rows
    ]


@app.post("/conversations", response_model=CreateConversationResponse)
async def create_conversation(principal: Principal = Depends(get_current_principal)) -> CreateConversationResponse:
    conversation_id = await chat_store.create_conversation(principal.subject)
    if conversation_id is None:
        raise HTTPException(status_code=503, detail="persistence not configured")
    return CreateConversationResponse(id=conversation_id)


@app.get("/conversations/{conversation_id}/messages", response_model=list[StoredMessage])
async def get_conversation_messages(
    conversation_id: str, principal: Principal = Depends(get_current_principal)
) -> list[StoredMessage]:
    if not chat_store.is_configured():
        return []
    if await chat_store.get_conversation(conversation_id, principal.subject) is None:
        raise HTTPException(status_code=404, detail="conversation not found")
    rows = await chat_store.list_messages(conversation_id, principal.subject)
    return [
        StoredMessage(
            id=str(r["id"]),
            role=r["role"],
            content=r["content"],
            audio_path=r["audio_path"],
            response_language=r["response_language"],
        )
        for r in rows
    ]


@app.delete("/conversations/{conversation_id}", status_code=204)
async def delete_conversation(conversation_id: str, principal: Principal = Depends(get_current_principal)) -> None:
    """Deletes the conversation and (via the schema's `on delete cascade`)
    every message in it. Does not delete any replay audio already uploaded
    to Supabase Storage for those messages -- out of scope here."""
    if not chat_store.is_configured():
        raise HTTPException(status_code=503, detail="persistence not configured")
    if not await chat_store.delete_conversation(conversation_id, principal.subject):
        raise HTTPException(status_code=404, detail="conversation not found")
