"""Hard gate for write-scope/irreversible tool actions triggered via voice
(S2S plan §6): a secondary typed/tapped confirmation is required before
execution — enforced in code, not left to a prompt instruction a model
could ignore or a fast talker could talk past.

`ConfirmationGate` issues a single-use token scoped to the exact tool name +
args it was requested for. A token can't be replayed for a different action
(even the same tool with different args), and once consumed it's gone.
task_agent.run_turn (see the write_scope_tools/confirmation_gate params)
never executes a gated tool from a voice-mode turn without a token that
passes `consume()` for that exact call.

Architecture gap closed: this used to be an in-memory-only, single-process
dict -- a confirmation issued on one instance was unredeemable on another,
silently re-prompting the user instead of executing an action they already
confirmed. Backed by Redis (one key per token, with a native TTL) when
REDIS_URL is configured, shared across every replica; falls back to the
exact previous in-memory behavior otherwise -- zero behavior change for a
single-instance deployment, same "optional at runtime" convention as
security/rate_limit.py and chat_store.py.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import time
from dataclasses import dataclass

import redis.asyncio as redis

# 5 minutes -- long enough for a real user to actually read and confirm/deny
# a write-scope action, short enough that (a) a token leaked via a log line
# or screen share doesn't stay exploitable indefinitely, and (b) abandoned
# confirmations (user never responds) don't accumulate forever in this
# process-lifetime singleton. Real gap caught in a pre-deploy sweep: tokens
# previously never expired at all.
_DEFAULT_TTL_SECONDS = 300.0


@dataclass
class PendingConfirmation:
    token: str
    tool_name: str
    args: dict


class ConfirmationGate:
    def __init__(self, *, ttl_seconds: float = _DEFAULT_TTL_SECONDS, redis_url_env: str = "REDIS_URL"):
        self._pending: dict[str, tuple[str, str, float]] = {}  # in-memory fallback: token -> (tool_name, args_hash, expires_at)
        self._ttl_seconds = ttl_seconds
        self._redis_url_env = redis_url_env
        self._redis: redis.Redis | None = None
        self._redis_checked = False

    def _get_redis(self) -> "redis.Redis | None":
        if not self._redis_checked:
            self._redis_checked = True
            url = os.environ.get(self._redis_url_env)
            if url:
                self._redis = redis.from_url(url)
        return self._redis

    @staticmethod
    def _args_hash(tool_name: str, args: dict) -> str:
        canonical = json.dumps({"tool": tool_name, "args": args}, sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()

    def _evict_expired(self) -> None:
        now = time.monotonic()
        expired = [token for token, (_, _, expires_at) in self._pending.items() if now >= expires_at]
        for token in expired:
            del self._pending[token]

    async def request_confirmation(self, tool_name: str, args: dict) -> PendingConfirmation:
        token = secrets.token_urlsafe(24)
        args_hash = self._args_hash(tool_name, args)
        r = self._get_redis()
        if r is not None:
            payload = json.dumps({"tool": tool_name, "hash": args_hash})
            await r.set(f"confirm:{token}", payload, ex=int(self._ttl_seconds))
        else:
            self._evict_expired()  # opportunistic -- bounds memory without a background task
            self._pending[token] = (tool_name, args_hash, time.monotonic() + self._ttl_seconds)
        return PendingConfirmation(token=token, tool_name=tool_name, args=args)

    async def consume(self, token: str, tool_name: str, args: dict) -> bool:
        """True and invalidates the token if it matches this exact
        tool+args and hasn't expired; False (no side effects, token
        untouched unless expired) otherwise — including any attempt to
        replay it for a different action."""
        r = self._get_redis()
        if r is not None:
            return await self._consume_redis(r, token, tool_name, args)
        return self._consume_in_memory(token, tool_name, args)

    async def _consume_redis(self, r: "redis.Redis", token: str, tool_name: str, args: dict) -> bool:
        raw = await r.get(f"confirm:{token}")
        if raw is None:
            return False  # never issued, already consumed, or expired (Redis's own TTL)
        entry = json.loads(raw)
        if entry["tool"] != tool_name or entry["hash"] != self._args_hash(tool_name, args):
            return False
        await r.delete(f"confirm:{token}")
        return True

    def _consume_in_memory(self, token: str, tool_name: str, args: dict) -> bool:
        entry = self._pending.get(token)
        if entry is None:
            return False
        expected_tool, expected_hash, expires_at = entry
        if time.monotonic() >= expires_at:
            del self._pending[token]
            return False
        if expected_tool != tool_name or expected_hash != self._args_hash(tool_name, args):
            return False
        del self._pending[token]
        return True
