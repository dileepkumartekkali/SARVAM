"""In-memory rate limiter for audio session creation (S2S plan §6: voice
sessions cost more than text turns, so the cost-exhaustion abuse surface is
bigger). Sliding-window counter, single-process — correct for one gateway
replica.

ponytail: in-memory + single-process, per-key deque. Swap for a Redis-backed
counter (already in docker-compose) before running more than one gateway
replica — counts here aren't shared across processes.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque


class SlidingWindowRateLimiter:
    def __init__(self, *, max_requests: int, window_seconds: float):
        self._max_requests = max_requests
        self._window_seconds = window_seconds
        self._hits: dict[str, deque] = defaultdict(deque)

    def allow(self, key: str, *, now: float | None = None) -> bool:
        now = now if now is not None else time.monotonic()
        hits = self._hits[key]
        cutoff = now - self._window_seconds
        while hits and hits[0] < cutoff:
            hits.popleft()
        if len(hits) >= self._max_requests:
            return False
        hits.append(now)
        return True
