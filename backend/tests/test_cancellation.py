import asyncio

import pytest

from agent_core.agents.cancellation import CancellationToken, TurnCancelled


async def test_run_returns_result_when_not_cancelled():
    token = CancellationToken()

    async def quick():
        return "done"

    result = await token.run(quick())
    assert result == "done"


async def test_run_raises_turn_cancelled_and_cancels_underlying_task_when_token_fires_first():
    token = CancellationToken()
    task_was_cancelled = False

    async def slow():
        nonlocal task_was_cancelled
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            task_was_cancelled = True
            raise

    async def cancel_soon():
        await asyncio.sleep(0.01)
        token.cancel()

    asyncio.ensure_future(cancel_soon())
    with pytest.raises(TurnCancelled):
        await token.run(slow())

    assert task_was_cancelled is True


def test_check_raises_only_after_cancel():
    token = CancellationToken()
    token.check()  # no-op, not cancelled yet
    token.cancel()
    with pytest.raises(TurnCancelled):
        token.check()
