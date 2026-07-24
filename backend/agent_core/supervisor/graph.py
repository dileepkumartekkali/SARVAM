"""Text→Text LangGraph wiring: language_agent -> task_agent, in-memory
checkpointer. On low-confidence language detection, the graph short-circuits
to a deterministic clarifying question instead of ever calling task_agent's
LLM — that call is never made, so there's no risk of it answering fluently in
the wrong language.
"""

from __future__ import annotations

from typing import Sequence, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from ..agents.language_agent import LOW_CONFIDENCE_THRESHOLD, detect_language
from ..agents.task_agent import CLARIFYING_QUESTION, ToolFn, run_turn
from ..agents.translation_policy import decide_translation
from ..llm_adapter.base import LLMRouter, ToolDefinition
from ..security.confirmation import ConfirmationGate, PendingConfirmation
from ..tools.rag_tool import TOOL_VERIFIED_MARKER
from .state import SessionState


class TextGraphState(TypedDict, total=False):
    session: SessionState
    user_message: str
    stt_language_hint: str | None
    confirmation_token: str | None
    response: str
    prompt_version: str
    tool_call_count: int
    self_check_ok: bool
    pending_confirmation: PendingConfirmation | None
    # Real bug hit live: task_agent_node returns "error" (line 142 below) but
    # this key was never declared here -- LangGraph silently discards writes
    # to channels not in this TypedDict (confirmed against the installed
    # langgraph==1.2.9 source), so `state.get("error")` in api/main.py's
    # /chat endpoint always saw the default False. A provider-failure apology
    # was persisted to chat_store as a real answer and reported to the
    # client as a success.
    error: bool
    # Prior turns of this conversation (plain user/assistant text), persisted
    # per thread_id by the checkpointer and fed back into task_agent — this is
    # what makes the agent stateful across turns instead of a stateless
    # LLM-call service that can't answer "what did I just ask you?".
    history: list


# ponytail: 20 messages (10 turns) keeps prompt size bounded; add rolling
# summarization if conversations outgrow it in practice.
_MAX_HISTORY_MESSAGES = 20


_UNSET = object()  # distinguishes "not passed" (default to `router`) from an explicit None


def build_text_graph(
    router: LLMRouter,
    *,
    tools: dict[str, ToolFn] | None = None,
    tool_definitions: Sequence[ToolDefinition] | None = None,
    tool_manifest: str = "",
    write_scope_tools: set[str] | None = None,
    max_tool_calls: int = 3,
    language_router: LLMRouter | None = _UNSET,
):
    # Defaults to the same router as task_agent — pass a distinct one (or
    # explicit None to disable the LLM fallback entirely) if language
    # classification should behave differently.
    language_router = router if language_router is _UNSET else language_router

    # One gate per compiled graph, not per turn — a confirmation issued on
    # turn N must still be redeemable on turn N+1. See its own single-process
    # caveat in security/confirmation.py / docs/THREAT_MODEL.md.
    confirmation_gate = ConfirmationGate() if write_scope_tools else None

    async def language_agent_node(state: TextGraphState) -> dict:
        result = await detect_language(
            state["user_message"], stt_hint=state.get("stt_language_hint"), router=language_router
        )
        session = state["session"].model_copy(
            update={
                "response_language": result.language,
                "language_confidence": result.confidence,
                "is_code_mixed": result.is_code_mixed,
                "translation_applied": decide_translation(result.language),
            }
        )
        return {"session": session}

    def route_after_language(state: TextGraphState) -> str:
        confidence = state["session"].language_confidence or 0.0
        return "clarify" if confidence < LOW_CONFIDENCE_THRESHOLD else "task_agent"

    async def clarify_node(state: TextGraphState) -> dict:
        return {
            "response": CLARIFYING_QUESTION,
            "prompt_version": "low_confidence_clarify",
            "tool_call_count": 0,
            "self_check_ok": True,
        }

    async def task_agent_node(state: TextGraphState) -> dict:
        history = list(state.get("history") or [])
        result = await run_turn(
            state["session"],
            router,
            state["user_message"],
            tools=tools,
            tool_definitions=tool_definitions,
            max_tool_calls=max_tool_calls,
            tool_manifest=tool_manifest,
            write_scope_tools=write_scope_tools,
            confirmation_gate=confirmation_gate,
            confirmation_token=state.get("confirmation_token"),
            history=history,
        )
        # The RAW user message goes into history (not the directive-prefixed
        # variant run_turn builds) — directives are per-turn steering, and
        # replaying stale ones from history would fight the current turn's.
        #
        # `result.error` (the provider-failure apology, not a real answer)
        # never gets appended here -- feeding it back as if it were a real
        # past turn would pollute the LLM's own context on the NEXT turn,
        # same reasoning as not persisting/speaking it anywhere else.
        # Marker appended ONLY in what the LLM sees back as its own history
        # next turn -- never in `result.text` (already returned/persisted
        # unmarked above). See rag_tool.TOOL_VERIFIED_MARKER: without this,
        # a wrong answer from a turn that skipped the tool got repeated
        # verbatim on a later question in the same conversation, since
        # history had no way to distinguish "checked" from "guessed."
        history_text = result.text
        if result.tool_call_count > 0:
            history_text = f"{result.text}\n{TOOL_VERIFIED_MARKER}"
        new_history = (
            history
            if result.error
            else history + [
                {"role": "user", "content": state["user_message"]},
                {"role": "assistant", "content": history_text},
            ]
        )
        return {
            "response": result.text,
            "prompt_version": result.prompt_version,
            "tool_call_count": result.tool_call_count,
            "self_check_ok": result.self_check_ok,
            "pending_confirmation": result.pending_confirmation,
            "history": new_history[-_MAX_HISTORY_MESSAGES:],
            "error": result.error,
        }

    graph = StateGraph(TextGraphState)
    graph.add_node("language_agent", language_agent_node)
    graph.add_node("task_agent", task_agent_node)
    graph.add_node("clarify", clarify_node)
    graph.set_entry_point("language_agent")
    graph.add_conditional_edges(
        "language_agent", route_after_language, {"clarify": "clarify", "task_agent": "task_agent"}
    )
    graph.add_edge("task_agent", END)
    graph.add_edge("clarify", END)
    return graph.compile(checkpointer=MemorySaver())
