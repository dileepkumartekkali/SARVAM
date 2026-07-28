"""A fake WebSocket connection for testing SarvamSTTClient/SarvamTTSClient
without a real socket — injected via each client's `connect` parameter,
mirroring the same DI pattern used for httpx `transport` in Phase 1."""

from __future__ import annotations

import asyncio


class FakeWSConnection:
    """Mimics the subset of `websockets`' client connection API the Sarvam
    adapters use: async context manager, `send()`, and async iteration over
    incoming messages."""

    def __init__(self, incoming=None, *, fail_with: Exception | None = None):
        self.incoming = list(incoming or [])
        self.sent: list = []
        self._fail_with = fail_with
        self._i = 0
        self._recv_in_progress = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def send(self, data) -> None:
        await asyncio.sleep(0)  # scheduling checkpoint — lets a concurrent pump task run
        self.sent.append(data)

    def __aiter__(self):
        return self

    async def __anext__(self):
        await asyncio.sleep(0)  # scheduling checkpoint — lets a concurrent pump task run
        if self._i >= len(self.incoming):
            if self._fail_with is not None:
                raise self._fail_with
            raise StopAsyncIteration
        item = self.incoming[self._i]
        self._i += 1
        return item

    async def recv(self):
        """`websockets`' single-message-at-a-time API (distinct from
        `__anext__`'s async-iterator protocol) — used by SarvamTTSClient's
        idle-timeout logic. When exhausted with no scripted failure, blocks
        forever rather than raising, simulating a real connection that's
        gone quiet (not closed) — exactly what lets a caller's
        `asyncio.wait_for(ws.recv(), timeout=...)` time out for real.

        Mirrors the real `websockets` library's own single-reader guard:
        raises the same RuntimeError a real connection does if a second
        recv() starts before the first one has actually finished (whether
        it returned, raised, or was cancelled) -- this is what makes a
        caller's `.cancel()`-without-awaiting bug (confirmed live: "cannot
        call recv while another coroutine is already running recv or
        recv_streaming") reproducible as a real, deterministic test
        failure instead of a live-only race."""
        if self._recv_in_progress:
            raise RuntimeError("cannot call recv while another coroutine is already running recv or recv_streaming")
        self._recv_in_progress = True
        try:
            await asyncio.sleep(0)
            if self._i >= len(self.incoming):
                if self._fail_with is not None:
                    raise self._fail_with
                await asyncio.Event().wait()  # never resolves — simulates a quiet-but-open socket
            item = self.incoming[self._i]
            self._i += 1
            return item
        finally:
            self._recv_in_progress = False


def fake_connect_returning(connection: FakeWSConnection):
    """Returns a `connect(url, **kwargs)`-shaped callable for DI into a client."""

    def _connect(url, **kwargs):
        return connection

    return _connect
