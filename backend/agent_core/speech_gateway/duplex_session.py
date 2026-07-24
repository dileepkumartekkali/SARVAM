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
        # Real bug hit live: main.py flips the phase to THINKING
        # synchronously, at the exact moment the whole-turn task is created
        # (closing a DIFFERENT race -- see that call site's own comment), but
        # `_task_agent_task`/`_tts_task` above are only set once run_turn's
        # own body actually executes far enough to reach start_thinking()/
        # start_speaking(). A barge-in landing in that gap (or in the gap
        # between think finishing and speak starting, e.g. while
        # `assistant_text` is being sent) saw phase=THINKING, reported
        # `barge_in_detected` to the client, but had nothing registered yet
        # to actually cancel -- the assistant kept right on thinking/speaking
        # over the user. Registering the OUTER task here, synchronously, at
        # the same call site as the phase flip, closes that gap: cancelling
        # it cascades into whatever it's currently awaiting (think_task,
        # speak_task, or neither), same as asyncio always cascades a
        # cancelled task's await chain.
        self._turn_task: asyncio.Task | None = None

    def register_turn_task(self, turn_task: asyncio.Task) -> None:
        """Called synchronously by the caller at the exact same point it
        flips the phase to THINKING and schedules the turn -- see this
        class's own `_turn_task` comment for why that synchronicity matters.
        """
        self._turn_task = turn_task

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
        self._turn_task = None
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
        # Cancel the whole-turn task FIRST -- it covers the gap windows
        # `_task_agent_task`/`_tts_task` alone can't (see `_turn_task`'s own
        # comment). Cancelling it cascades into whatever it's currently
        # awaiting, so by the time this returns, `_task_agent_task`/
        # `_tts_task` (if either was ever registered) are already done —
        # the calls below become harmless no-ops via `_cancel_and_await`'s
        # own `task.done()` check, kept as a second layer of defense rather
        # than removed.
        await self._cancel_and_await(self._turn_task)
        await self._cancel_and_await(self._task_agent_task)
        await self._cancel_and_await(self._tts_task)  # closes the TTS socket via __aexit__

        self._task_agent_task = None
        self._tts_task = None
        self._turn_task = None
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
