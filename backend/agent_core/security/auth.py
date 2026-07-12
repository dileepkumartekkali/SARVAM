"""JWT verification + RBAC — the API Gateway's access-control layer.

Signature/expiry verification uses PyJWT (never hand-rolled — see
pyproject.toml comment). This module is the **resource-server** side of
OAuth2: it verifies tokens an identity provider (Auth0/Okta/Cognito/etc.)
issued. It does not issue tokens and does not implement an authorization-code
exchange — that's a deployment-time IdP integration, configured via
`AuthConfig` (issuer/audience/secret or JWKS in production), not code that
belongs in this repo. Wiring a real IdP should only ever mean changing env
vars, never this file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

_bearer_scheme = HTTPBearer(auto_error=False)


@dataclass
class AuthConfig:
    secret_env: str = "JWT_SIGNING_SECRET"
    algorithm: str = "HS256"
    audience: str | None = None
    issuer: str | None = None

    def secret(self) -> str:
        secret = os.environ.get(self.secret_env)
        if not secret:
            raise RuntimeError(f"{self.secret_env} not set")
        return secret


class AuthError(Exception):
    def __init__(self, message: str, *, status_code: int = status.HTTP_401_UNAUTHORIZED):
        super().__init__(message)
        self.status_code = status_code


@dataclass
class Principal:
    subject: str
    roles: frozenset = field(default_factory=frozenset)
    claims: dict = field(default_factory=dict)

    def has_role(self, role: str) -> bool:
        return role in self.roles


def decode_token(token: str, config: AuthConfig | None = None) -> Principal:
    cfg = config or AuthConfig()
    try:
        secret = cfg.secret()
    except RuntimeError as e:
        # A real bug hit in production: a missing JWT_SIGNING_SECRET raised
        # here, uncaught, as a bare RuntimeError — every authenticated
        # request (including /chat) crashed with an unhandled-exception 500
        # and no diagnosable message. This is a server misconfiguration, not
        # a client auth failure, so it stays a 500 — but now with a message
        # that actually says what's missing, in logs and the response body.
        raise AuthError(f"server misconfigured: {e}", status_code=status.HTTP_500_INTERNAL_SERVER_ERROR) from e
    try:
        claims = jwt.decode(
            token,
            secret,
            algorithms=[cfg.algorithm],
            audience=cfg.audience,
            issuer=cfg.issuer,
            options={"require": ["exp", "sub"], "verify_aud": cfg.audience is not None},
        )
    except jwt.ExpiredSignatureError as e:
        raise AuthError("token expired") from e
    except jwt.InvalidTokenError as e:
        raise AuthError(f"invalid token: {e}") from e

    return Principal(subject=claims["sub"], roles=frozenset(claims.get("roles", [])), claims=claims)


async def get_current_principal(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> Principal:
    """FastAPI dependency: 401 on missing/invalid/expired bearer token.

    `JWT_AUDIENCE` is optional (unset = audience unchecked, same as before) —
    set it to "authenticated" when the IdP is Supabase, which stamps that
    exact value into every token's `aud` claim.
    """
    if credentials is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="missing bearer token")
    try:
        return decode_token(credentials.credentials, AuthConfig(audience=os.environ.get("JWT_AUDIENCE")))
    except AuthError as e:
        raise HTTPException(e.status_code, detail=str(e)) from e


def require_role(role: str):
    """RBAC gate for write-scope tools/endpoints — 403 if the principal
    lacks the role. This is the enforcement point; a tool's own code never
    has to remember to check permissions itself."""

    async def _dependency(principal: Principal = Depends(get_current_principal)) -> Principal:
        if not principal.has_role(role):
            raise HTTPException(status.HTTP_403_FORBIDDEN, detail=f"requires role: {role}")
        return principal

    return _dependency
