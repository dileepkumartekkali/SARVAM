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

from ._fakes import ScriptedProvider

_REQUEST_BODY = {
    "session_id": "s1",
    "conversation_id": "c1",
    "thread_id": "t1",
    "message": "hi there",
}


def _auth_header(secret="test-secret", exp_delta=3600):
    token = jwt.encode({"sub": "user-1", "exp": int(time.time()) + exp_delta}, secret, algorithm="HS256")
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
