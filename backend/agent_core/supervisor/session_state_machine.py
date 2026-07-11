"""Session state machine for full-duplex Speech→Speech (Phase 5).

Three phases: LISTENING (mic active, no turn in flight), THINKING (task_agent
reasoning, possibly calling tools), SPEAKING (TTS playback). The one hard
rule from the S2S plan §3: barge-in must re-enter LISTENING from ANY state —
a rapid double barge-in (interrupting an interruption's own recovery) must
not get stuck or throw, it just resets again. `barge_in()` is idempotent for
exactly that reason.
"""

from __future__ import annotations

from enum import Enum


class SessionPhase(str, Enum):
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"


class SessionStateMachine:
    def __init__(self):
        self.phase = SessionPhase.LISTENING

    def transition_to(self, phase: SessionPhase) -> None:
        self.phase = phase

    def barge_in(self) -> bool:
        """Re-enters LISTENING from any phase. Returns True if this actually
        interrupted something in progress (THINKING/SPEAKING); False if the
        session was already idle — callers use this to decide whether a
        `barge_in_detected` event is even meaningful to emit.
        """
        was_active = self.phase in (SessionPhase.THINKING, SessionPhase.SPEAKING)
        self.phase = SessionPhase.LISTENING
        return was_active
