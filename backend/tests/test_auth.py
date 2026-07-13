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


def test_supabase_jwks_verified_token_decodes_to_principal(monkeypatch):
    """Supabase's current default: JWTs are signed asymmetrically (ES256),
    not with the shared HS256 secret — verified live against a real project
    (a real access token's header carried `alg: ES256`). Mocks the JWKS fetch
    (no network in tests) but exercises the real ES256 verify path."""
    from cryptography.hazmat.primitives.asymmetric import ec

    import agent_core.security.auth as auth_module

    private_key = ec.generate_private_key(ec.SECP256R1())
    token = jwt.encode(
        {"sub": "supabase-user-1", "exp": int(time.time()) + 3600, "aud": "authenticated"},
        private_key,
        algorithm="ES256",
        headers={"kid": "test-kid"},
    )

    class _FakeSigningKey:
        key = private_key.public_key()

    class _FakeJWKClient:
        def get_signing_key_from_jwt(self, _token):
            return _FakeSigningKey()

    monkeypatch.setenv("SUPABASE_URL", "https://fake-project.supabase.co")
    auth_module._jwks_client.cache_clear()
    monkeypatch.setattr(auth_module, "_jwks_client", lambda url: _FakeJWKClient())

    principal = decode_token(token, AuthConfig(audience="authenticated"))

    assert principal.subject == "supabase-user-1"


def test_unreachable_jwks_is_a_clean_500(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://fake-project.supabase.co")

    import agent_core.security.auth as auth_module

    def _raise(_url):
        raise jwt.PyJWKClientError("could not fetch JWKS")

    auth_module._jwks_client.cache_clear()
    monkeypatch.setattr(auth_module, "_jwks_client", _raise)

    with pytest.raises(AuthError) as exc_info:
        decode_token(_make_token())

    assert exc_info.value.status_code == 500
