from fastapi.testclient import TestClient

from agent_core.api.main import app
from agent_core.security.auth import decode_token


def test_dev_login_disabled_by_default(monkeypatch):
    monkeypatch.delenv("DEV_AUTH_ENABLED", raising=False)
    client = TestClient(app)
    resp = client.post("/auth/dev-login", json={"username": "alice"})
    assert resp.status_code == 404


def test_dev_login_issues_valid_token_when_enabled(monkeypatch):
    monkeypatch.setenv("DEV_AUTH_ENABLED", "true")
    monkeypatch.setenv("JWT_SIGNING_SECRET", "test-secret")
    client = TestClient(app)

    resp = client.post("/auth/dev-login", json={"username": "alice"})

    assert resp.status_code == 200
    token = resp.json()["access_token"]
    principal = decode_token(token)
    assert principal.subject == "alice"
    assert principal.has_role("dev_user")
