"""Full-duplex Speech→Speech session (Phase 5): STT stays open during TTS
playback, so a `speech_start` VAD signal while THINKING/SPEAKING is a
barge-in. Implements the full interrupt sequence from S2S plan §3:

    stop local playback (client) -> close the active TTS socket (gateway,
    discarding in-flight chunks — expected) -> cancel the in-flight
    task_agent call -> discard any unspoken response text -> emit
    barge_in_detected -> re-enter LISTENING

`handle_vad_speech_start()` is idempotent and safe to call from ANY state,
including immediately after a previous call — a rapid double barge-in
(interrupting an interruption's own recovery) must not orphan a task or
throw. See tests/test_duplex_session.py::test_rapid_double_barge_in.
"""

from __future__ import annotations

import asyncio

from ..agents.cancellation import CancellationToken
from ..supervisor.session_state_machine import SessionPhase, SessionStateMachine


class DuplexSession:
    def __init__(self):
        self.state = SessionStateMachine()
        self.cancellation_token = CancellationToken()
        self.events: list[dict] = []
        self._task_agent_task: asyncio.Task | None = None
        self._tts_task: asyncio.Task | None = None

    def start_thinking(self, task_agent_task: asyncio.Task) -> None:
        self.state.transition_to(SessionPhase.THINKING)
        self._task_agent_task = task_agent_task

    def start_speaking(self, tts_task: asyncio.Task) -> None:
        self.state.transition_to(SessionPhase.SPEAKING)
        self._tts_task = tts_task

    def finish_turn(self) -> None:
        """Normal (non-interrupted) turn completion."""
        self._task_agent_task = None
        self._tts_task = None
        self.cancellation_token = CancellationToken()
        self.state.transition_to(SessionPhase.LISTENING)

    async def handle_vad_speech_start(self) -> bool:
        """Returns True if this call performed a real barge-in (something was
        interrupted); False if the session was already idle.
        """
        was_active = self.state.barge_in()  # re-enters LISTENING regardless
        if not was_active:
            return False

        self.cancellation_token.cancel()
        await self._cancel_and_await(self._task_agent_task)
        await self._cancel_and_await(self._tts_task)  # closes the TTS socket via __aexit__

        self._task_agent_task = None
        self._tts_task = None
        self.cancellation_token = CancellationToken()  # fresh token for the next turn
        self.events.append({"type": "barge_in_detected"})
        return True

    @staticmethod
    async def _cancel_and_await(task: asyncio.Task | None) -> None:
        if task is None or task.done():
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception:
            pass  # discarding an interrupted task's error — we're throwing its result away anyway
