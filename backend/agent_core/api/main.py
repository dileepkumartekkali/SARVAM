"""FastAPI app entrypoint.

Text→Text chat via LangGraph, real providers configured from env. Voice
WebSocket routes live in the Speech Gateway (its own service).
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ..agents.language_agent import LOW_CONFIDENCE_THRESHOLD, detect_language
from ..agents.task_agent import CLARIFYING_QUESTION, stream_turn
from ..agents.translation_policy import decide_translation
from ..llm_adapter import build_router_from_env
from ..observability.logging_config import configure_logging
from ..observability.metrics import metrics_response
from ..observability.tracing import init_tracing
from ..persistence import chat_store
from ..security.auth import Principal, get_current_principal
from ..security.confirmation import ConfirmationGate
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
    allow_methods=["GET", "POST", "DELETE"],
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

# /chat/stream bypasses the graph (task_agent.stream_turn is called directly —
# see its own module for why), so it can't use the graph's LangGraph
# checkpointer for cross-turn history. This is its own equivalent, same
# per-thread_id/cap-at-20 shape as supervisor/graph.py's _MAX_HISTORY_MESSAGES,
# kept independent since production traffic uses one endpoint exclusively
# (the other stays as a non-streaming reference for load_test.py/chaos_test.py).
_STREAM_MAX_HISTORY_MESSAGES = 20
_stream_history: dict[str, list[dict]] = {}
# One gate for the process, not per-request — a confirmation issued on turn N
# must still be redeemable on turn N+1 (same reasoning as build_text_graph's).
_stream_confirmation_gate = ConfirmationGate() if _tool_registry.write_scope_names() else None


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


def _sse_event(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest, principal: Principal = Depends(get_current_principal)) -> StreamingResponse:
    """Streams `{"type": "text_delta", "text": ...}` events as sentence-bounded
    chunks become ready (task_agent.stream_turn), then one final `{"type":
    "done", ...}` event shaped like ChatResponse's fields. See stream_turn's
    own docstring for the accepted trade-offs (no correction-retry once
    streaming has started; legacy TOOL_CALL: text convention only, not
    native, for tool detection). The non-streaming `/chat` above is
    untouched and still used by scripts/load_test.py and chaos_test.py.
    """
    persistence_on = chat_store.is_configured()
    if persistence_on and await chat_store.get_conversation(req.conversation_id, principal.subject) is None:
        raise HTTPException(status_code=404, detail="conversation not found")

    async def event_stream():
        session = SessionState(
            session_id=req.session_id,
            conversation_id=req.conversation_id,
            thread_id=req.thread_id,
            mode=Mode(req.mode),
        )
        lang_result = await detect_language(req.message, stt_hint=req.stt_language_hint, router=_router)
        session = session.model_copy(
            update={
                "response_language": lang_result.language,
                "language_confidence": lang_result.confidence,
                "is_code_mixed": lang_result.is_code_mixed,
                "translation_applied": decide_translation(lang_result.language),
            }
        )

        # Mirrors graph.py's route_after_language — never call the LLM at all
        # on low-confidence input, ask a deterministic clarifying question.
        if (lang_result.confidence or 0.0) < LOW_CONFIDENCE_THRESHOLD:
            final_text = CLARIFYING_QUESTION
            self_check_ok = True
            pending = None
            yield _sse_event({"type": "text_delta", "text": final_text})
        else:
            history = _stream_history.get(req.thread_id, [])
            final_event = None
            async for event in stream_turn(
                session,
                _router,
                req.message,
                tools=_tool_registry.as_dispatch_dict(),
                tool_definitions=_tool_registry.as_tool_definitions(),
                tool_manifest=_tool_registry.as_prompt_manifest(),
                write_scope_tools=_tool_registry.write_scope_names(),
                confirmation_gate=_stream_confirmation_gate,
                confirmation_token=req.confirmation_token,
                history=history,
            ):
                if event["type"] == "text_delta":
                    yield _sse_event(event)
                else:
                    final_event = event
            final_text = final_event["text"]
            self_check_ok = final_event["self_check_ok"]
            pending = final_event.get("pending_confirmation")
            # Clarify turns are deliberately NOT added to history, matching
            # graph.py's clarify_node (which never touches the "history" key).
            _stream_history[req.thread_id] = (
                history + [{"role": "user", "content": req.message}, {"role": "assistant", "content": final_text}]
            )[-_STREAM_MAX_HISTORY_MESSAGES:]

        assistant_message_id = None
        if persistence_on:
            assistant_message_id = str(uuid.uuid4())
            asyncio.create_task(
                chat_store.record_turn(
                    req.conversation_id,
                    principal.subject,
                    req.message,
                    final_text,
                    session.response_language,
                    assistant_message_id,
                )
            )

        yield _sse_event(
            {
                "type": "done",
                "message_id": assistant_message_id,
                "response_language": session.response_language,
                "language_confidence": session.language_confidence,
                "is_code_mixed": session.is_code_mixed,
                "self_check_ok": self_check_ok,
                "pending_confirmation": (
                    {"token": pending.token, "tool_name": pending.tool_name, "args": pending.args}
                    if pending is not None
                    else None
                ),
            }
        )

    return StreamingResponse(event_stream(), media_type="text/event-stream")


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
