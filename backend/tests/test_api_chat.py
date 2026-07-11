"""POST /chat end-to-end via FastAPI's TestClient — the "simple REST call".

The real router (built from env at import time) is swapped for a fake-provider
graph via FastAPI's dependency-override mechanism, so this runs with no
network access and no API keys. Auth is real (PyJWT signature verification
against a real token) for the happy-path test, and the two rejection tests
prove the endpoint is not reachable without one.
"""

import time

import jwt
from fastapi.testclient import TestClient

from agent_core.api.main import app, get_graph
from agent_core.llm_adapter import LLMRouter
from agent_core.supervisor.graph import build_text_graph

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


def test_chat_endpoint_end_to_end(monkeypatch):
    monkeypatch.setenv("JWT_SIGNING_SECRET", "test-secret")
    provider = ScriptedProvider(["Hello there, this is a plain text reply.", "OK"])
    fake_graph = build_text_graph(LLMRouter([provider]))
    app.dependency_overrides[get_graph] = lambda: fake_graph
    try:
        client = TestClient(app)
        resp = client.post("/chat", json=_REQUEST_BODY, headers=_auth_header())
        assert resp.status_code == 200
        body = resp.json()
        assert body["response"] == "Hello there, this is a plain text reply."
        assert body["prompt_version"] == "text_mode_system.v1"
        assert body["tool_call_count"] == 0
        assert body["response_language"] == "en"  # "hi" is an English-stopword match
    finally:
        app.dependency_overrides.clear()


def test_chat_endpoint_rejects_missing_auth():
    client = TestClient(app)
    resp = client.post("/chat", json=_REQUEST_BODY)
    assert resp.status_code == 401


def test_chat_endpoint_rejects_expired_token(monkeypatch):
    monkeypatch.setenv("JWT_SIGNING_SECRET", "test-secret")
    client = TestClient(app)
    resp = client.post("/chat", json=_REQUEST_BODY, headers=_auth_header(exp_delta=-10))
    assert resp.status_code == 401
