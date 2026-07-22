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
"""

from __future__ import annotations

import hashlib
import json
import secrets
import time
from dataclasses import dataclass

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
    def __init__(self, *, ttl_seconds: float = _DEFAULT_TTL_SECONDS):
        self._pending: dict[str, tuple[str, str, float]] = {}  # token -> (tool_name, args_hash, expires_at)
        self._ttl_seconds = ttl_seconds

    @staticmethod
    def _args_hash(tool_name: str, args: dict) -> str:
        canonical = json.dumps({"tool": tool_name, "args": args}, sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()

    def _evict_expired(self) -> None:
        now = time.monotonic()
        expired = [token for token, (_, _, expires_at) in self._pending.items() if now >= expires_at]
        for token in expired:
            del self._pending[token]

    def request_confirmation(self, tool_name: str, args: dict) -> PendingConfirmation:
        self._evict_expired()  # opportunistic -- bounds memory without a background task
        token = secrets.token_urlsafe(24)
        self._pending[token] = (tool_name, self._args_hash(tool_name, args), time.monotonic() + self._ttl_seconds)
        return PendingConfirmation(token=token, tool_name=tool_name, args=args)

    def consume(self, token: str, tool_name: str, args: dict) -> bool:
        """True and invalidates the token if it matches this exact
        tool+args and hasn't expired; False (no side effects, token
        untouched unless expired) otherwise — including any attempt to
        replay it for a different action."""
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
