import time

import jwt
import pytest

from agent_core.security.auth import AuthConfig, AuthError, Principal, decode_token


def _make_token(*, secret="test-secret", sub="user-1", roles=None, exp_delta=3600):
    payload = {"sub": sub, "exp": int(time.time()) + exp_delta}
    if roles is not None:
        payload["roles"] = roles
    return jwt.encode(payload, secret, algorithm="HS256")


def test_missing_signing_secret_is_a_clean_500_not_an_unhandled_crash(monkeypatch):
    """Real bug hit in production: a missing JWT_SIGNING_SECRET on the
    deployed server raised a bare, uncaught RuntimeError from
    AuthConfig.secret() — every authenticated request (including /chat)
    crashed with an unhandled-exception 500 and no diagnosable message."""
    monkeypatch.delenv("JWT_SIGNING_SECRET", raising=False)
    token = _make_token()

    with pytest.raises(AuthError) as exc_info:
        decode_token(token)

    assert exc_info.value.status_code == 500
    assert "JWT_SIGNING_SECRET" in str(exc_info.value)


def test_valid_token_decodes_to_principal(monkeypatch):
    monkeypatch.setenv("JWT_SIGNING_SECRET", "test-secret")
    token = _make_token(roles=["admin", "voice_write"])

    principal = decode_token(token)

    assert isinstance(principal, Principal)
    assert principal.subject == "user-1"
    assert principal.has_role("admin")
    assert principal.has_role("voice_write")
    assert not principal.has_role("nonexistent")


def test_expired_token_rejected(monkeypatch):
    monkeypatch.setenv("JWT_SIGNING_SECRET", "test-secret")
    token = _make_token(exp_delta=-10)

    with pytest.raises(AuthError):
        decode_token(token)


def test_wrong_signature_rejected(monkeypatch):
    monkeypatch.setenv("JWT_SIGNING_SECRET", "test-secret")
    token = _make_token(secret="wrong-secret")

    with pytest.raises(AuthError):
        decode_token(token)


def test_missing_sub_claim_rejected(monkeypatch):
    monkeypatch.setenv("JWT_SIGNING_SECRET", "test-secret")
    token = jwt.encode({"exp": int(time.time()) + 3600}, "test-secret", algorithm="HS256")

    with pytest.raises(AuthError):
        decode_token(token)


def test_no_roles_claim_defaults_to_empty(monkeypatch):
    monkeypatch.setenv("JWT_SIGNING_SECRET", "test-secret")
    token = _make_token(roles=None)

    principal = decode_token(token)

    assert principal.roles == frozenset()
    assert not principal.has_role("admin")


def test_audience_mismatch_rejected(monkeypatch):
    monkeypatch.setenv("JWT_SIGNING_SECRET", "test-secret")
    token = jwt.encode(
        {"sub": "user-1", "exp": int(time.time()) + 3600, "aud": "other-audience"},
        "test-secret",
        algorithm="HS256",
    )

    with pytest.raises(AuthError):
        decode_token(token, AuthConfig(audience="expected-audience"))
