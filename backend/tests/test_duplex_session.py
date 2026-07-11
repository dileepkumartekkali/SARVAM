"""DuplexSession: single barge-in, and the explicit rapid-double-barge-in
case the task called out as "where most voice agents silently break" —
interrupting an interruption's own recovery must not orphan a task or throw.
"""

import asyncio

import pytest

from agent_core.speech_gateway.duplex_session import DuplexSession
from agent_core.supervisor.session_state_machine import SessionPhase


async def _never_ending():
    try:
        await asyncio.sleep(30)
    except asyncio.CancelledError:
        raise


async def test_barge_in_while_listening_is_a_no_op():
    session = DuplexSession()

    was_barge_in = await session.handle_vad_speech_start()

    assert was_barge_in is False
    assert session.events == []
    assert session.state.phase == SessionPhase.LISTENING


async def test_single_barge_in_cancels_task_agent_and_tts_and_emits_event():
    session = DuplexSession()
    task_agent_task = asyncio.ensure_future(_never_ending())
    session.start_thinking(task_agent_task)
    tts_task = asyncio.ensure_future(_never_ending())
    session.start_speaking(tts_task)

    was_barge_in = await session.handle_vad_speech_start()

    assert was_barge_in is True
    assert task_agent_task.cancelled()
    assert tts_task.cancelled()
    assert session.events == [{"type": "barge_in_detected"}]
    assert session.state.phase == SessionPhase.LISTENING
    # no dangling references to the cancelled tasks — nothing "in flight" left behind
    assert session._task_agent_task is None
    assert session._tts_task is None


async def test_rapid_double_barge_in_does_not_orphan_a_task_or_throw():
    """Barge in once, immediately start a new turn, barge in again before
    that turn's tasks ever get a chance to run to completion — the exact
    "interrupting an interruption" case."""
    session = DuplexSession()

    first_task_agent = asyncio.ensure_future(_never_ending())
    first_tts = asyncio.ensure_future(_never_ending())
    session.start_thinking(first_task_agent)
    session.start_speaking(first_tts)

    assert await session.handle_vad_speech_start() is True  # barge-in #1

    # A new turn starts immediately (as the graph would, on the fresh transcript).
    second_task_agent = asyncio.ensure_future(_never_ending())
    session.start_thinking(second_task_agent)
    second_tts = asyncio.ensure_future(_never_ending())
    session.start_speaking(second_tts)

    assert await session.handle_vad_speech_start() is True  # barge-in #2, immediately

    # Every task ever tracked is actually cancelled — none left running.
    for task in (first_task_agent, first_tts, second_task_agent, second_tts):
        assert task.cancelled() or task.done()

    assert session.state.phase == SessionPhase.LISTENING
    assert session._task_agent_task is None
    assert session._tts_task is None
    # Two distinct barge-in events, not deduplicated away or dropped.
    assert session.events == [{"type": "barge_in_detected"}, {"type": "barge_in_detected"}]


async def test_triple_rapid_barge_in_from_idle_between_each_is_safe():
    session = DuplexSession()
    for _ in range(3):
        t = asyncio.ensure_future(_never_ending())
        session.start_thinking(t)
        assert await session.handle_vad_speech_start() is True
        # immediately barge in again while already idle — must be a safe no-op
        assert await session.handle_vad_speech_start() is False
        assert t.cancelled()

    assert session.state.phase == SessionPhase.LISTENING


async def test_finish_turn_resets_cleanly_for_a_normal_non_interrupted_turn():
    session = DuplexSession()
    t = asyncio.ensure_future(_never_ending())
    session.start_thinking(t)
    t.cancel()
    try:
        await t
    except asyncio.CancelledError:
        pass

    session.finish_turn()

    assert session.state.phase == SessionPhase.LISTENING
    assert session.events == []  # normal completion isn't a barge-in
