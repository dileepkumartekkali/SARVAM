"""Cooperative cancellation for task_agent's reasoning loop.

"Stop reading the stream" only interrupts a generator the caller is actively
pulling from — it does nothing about an in-flight LLM call the loop is
`await`-ing, or a tool call already dispatched. A `CancellationToken` is
threaded through `run_turn` explicitly (see task_agent.py) so a barge-in can
actually cancel the underlying asyncio task, not just stop consuming its
eventual result.
"""

from __future__ import annotations

import asyncio


class TurnCancelled(Exception):
    """Raised when a cancellation token fires mid-turn."""


class CancellationToken:
    def __init__(self):
        self._event = asyncio.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def is_cancelled(self) -> bool:
        return self._event.is_set()

    def check(self) -> None:
        """Raises TurnCancelled if already cancelled — call before starting
        any new step (a tool call, the next LLM call) so cancellation is
        never missed between checks."""
        if self.is_cancelled:
            raise TurnCancelled()

    async def run(self, coro):
        """Runs `coro`, actually cancelling its underlying task the instant
        the token fires — not just discarding its eventual result. This is
        what makes an in-flight LLM call abortable rather than merely
        ignorable.
        """
        self.check()
        task = asyncio.ensure_future(coro)
        cancel_wait = asyncio.ensure_future(self._event.wait())
        done, _ = await asyncio.wait({task, cancel_wait}, return_when=asyncio.FIRST_COMPLETED)
        if task in done:
            cancel_wait.cancel()
            return task.result()

        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass
        raise TurnCancelled()
