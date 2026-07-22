"""POST /chat/stream — SSE streaming endpoint, via FastAPI's TestClient (which
fully drains a StreamingResponse's body, so this can parse it like a real SSE
client would). This endpoint doesn't use `get_graph()`/dependency_overrides
(it calls task_agent.stream_turn directly, not the graph — see main.py's own
docstring on it), so the module-level `_router` is monkeypatched directly.
"""

import json
import time

import jwt
from fastapi.testclient import TestClient

import agent_core.api.main as main_module
from agent_core.api.main import app
from agent_core.llm_adapter import LLMRouter
from agent_core.tools.rag_tool import TOOL_VERIFIED_MARKER

from ._fakes import ScriptedProvider

_REQUEST_BODY = {
    "session_id": "s1",
    "conversation_id": "c1",
    "thread_id": "t1",
    "message": "hi there",
}


def _auth_header(secret="test-secret", exp_delta=3600, subject="user-1"):
    token = jwt.encode({"sub": subject, "exp": int(time.time()) + exp_delta}, secret, algorithm="HS256")
    return {"Authorization": f"Bearer {token}"}


def _parse_sse(text):
    events = []
    for block in text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        for line in block.splitlines():
            if line.startswith("data:"):
                events.append(json.loads(line[len("data:") :].strip()))
    return events


def test_chat_stream_end_to_end(monkeypatch):
    monkeypatch.setenv("JWT_SIGNING_SECRET", "test-secret")
    provider = ScriptedProvider(["Hello there, this is a plain text reply.", "OK"])
    monkeypatch.setattr(main_module, "_router", LLMRouter([provider]))

    client = TestClient(app)
    resp = client.post("/chat/stream", json=_REQUEST_BODY, headers=_auth_header())

    assert resp.status_code == 200
    events = _parse_sse(resp.text)
    assert len(events) >= 3  # language, at least one text_delta, final done
    # The detected language arrives BEFORE any text_delta -- the frontend
    # needs it to open its TTS socket with the right voice from chunk one,
    # not after the whole reply (and its language) is already known.
    assert events[0]["type"] == "language"
    assert events[0]["response_language"] == "en"
    done = events[-1]
    assert done["type"] == "done"
    assert done["self_check_ok"] is True
    assert done["response_language"] == "en"
    combined = "".join(e["text"] for e in events if e["type"] == "text_delta")
    assert "Hello there" in combined


def test_chat_stream_rejects_missing_auth():
    client = TestClient(app)
    resp = client.post("/chat/stream", json=_REQUEST_BODY)
    assert resp.status_code == 401


def test_chat_stream_low_confidence_never_reaches_the_answer_llm(monkeypatch):
    """Ambiguous input must never reach task_agent's real answer-generating
    call -- only the language-classify call (call #1) may fire."""
    monkeypatch.setenv("JWT_SIGNING_SECRET", "test-secret")
    provider = ScriptedProvider(['{"language": "en", "confidence": 0.2, "is_code_mixed": false}'])
    monkeypatch.setattr(main_module, "_router", LLMRouter([provider]))

    body = {**_REQUEST_BODY, "message": "hmm"}
    client = TestClient(app)
    resp = client.post("/chat/stream", json=body, headers=_auth_header())

    events = _parse_sse(resp.text)
    done = events[-1]
    assert done["type"] == "done"
    assert done["self_check_ok"] is True
    combined = "".join(e["text"] for e in events if e["type"] == "text_delta")
    assert "?" in combined  # the deterministic clarifying question
    assert provider.call_count == 1


def test_chat_stream_history_is_isolated_per_user_not_just_thread_id(monkeypatch):
    """Real gap caught in a pre-deploy sweep: _stream_history used to be
    keyed by thread_id alone -- conversation_id is ownership-checked, but
    thread_id is a separate client-supplied field that wasn't. Two
    different authenticated users sending the SAME thread_id must never
    see each other's history."""
    monkeypatch.setenv("JWT_SIGNING_SECRET", "test-secret")
    provider = ScriptedProvider(
        ["Nice to meet you, Alice.", "OK", "I don't know your name.", "OK"]
    )
    monkeypatch.setattr(main_module, "_router", LLMRouter([provider]))
    client = TestClient(app)

    shared_body = {**_REQUEST_BODY, "thread_id": "shared-thread"}
    client.post(
        "/chat/stream",
        json={**shared_body, "message": "my name is Alice"},
        headers=_auth_header(subject="user-1"),
    )

    client.post(
        "/chat/stream",
        json={**shared_body, "message": "what is my name"},
        headers=_auth_header(subject="user-2"),
    )

    # user-2's call must not have Alice's turn anywhere in its messages --
    # the SAME thread_id belonging to a different user is a fresh history.
    second_call_messages = provider.messages_by_call[2]
    combined = " ".join(str(m["content"]) for m in second_call_messages)
    assert "Alice" not in combined


def test_chat_stream_history_marks_tool_verified_turns_only(monkeypatch):
    """Real bug hit live: a wrong answer from a turn that skipped a tool got
    repeated verbatim on a later question in the same conversation --
    history had no way to tell "checked" from "guessed." A tool-using
    turn's stored history entry must carry TOOL_VERIFIED_MARKER; a plain
    turn's must not."""
    monkeypatch.setenv("JWT_SIGNING_SECRET", "test-secret")
    provider = ScriptedProvider(
        [
            'TOOL_CALL: {"name": "get_current_datetime", "args": {}}',  # sniffed, discarded
            'TOOL_CALL: {"name": "get_current_datetime", "args": {}}',  # run_turn's own dispatch
            "It is currently daytime.",  # run_turn's final answer
            "OK",  # self-check
            "Just a plain reply, no tool needed.",  # second turn, no tool call
            "OK",
        ]
    )
    monkeypatch.setattr(main_module, "_router", LLMRouter([provider]))
    client = TestClient(app)

    tool_body = {**_REQUEST_BODY, "thread_id": "marker-test-thread", "message": "what time is it"}
    client.post("/chat/stream", json=tool_body, headers=_auth_header(subject="marker-user"))

    plain_body = {**_REQUEST_BODY, "thread_id": "marker-test-thread", "message": "thanks"}
    client.post("/chat/stream", json=plain_body, headers=_auth_header(subject="marker-user"))

    history = main_module._stream_history[("marker-user", "marker-test-thread")]
    assistant_entries = [m["content"] for m in history if m["role"] == "assistant"]
    assert TOOL_VERIFIED_MARKER in assistant_entries[0]  # the tool-using turn
    assert TOOL_VERIFIED_MARKER not in assistant_entries[1]  # the plain turn
