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
from dataclasses import dataclass


@dataclass
class PendingConfirmation:
    token: str
    tool_name: str
    args: dict


class ConfirmationGate:
    def __init__(self):
        self._pending: dict[str, tuple[str, str]] = {}  # token -> (tool_name, args_hash)

    @staticmethod
    def _args_hash(tool_name: str, args: dict) -> str:
        canonical = json.dumps({"tool": tool_name, "args": args}, sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()

    def request_confirmation(self, tool_name: str, args: dict) -> PendingConfirmation:
        token = secrets.token_urlsafe(24)
        self._pending[token] = (tool_name, self._args_hash(tool_name, args))
        return PendingConfirmation(token=token, tool_name=tool_name, args=args)

    def consume(self, token: str, tool_name: str, args: dict) -> bool:
        """True and invalidates the token if it matches this exact
        tool+args; False (no side effects, token untouched) otherwise —
        including any attempt to replay it for a different action."""
        entry = self._pending.get(token)
        if entry is None:
            return False
        expected_tool, expected_hash = entry
        if expected_tool != tool_name or expected_hash != self._args_hash(tool_name, args):
            return False
        del self._pending[token]
        return True
