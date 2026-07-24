"""Rate limiter for audio session creation (S2S plan §6: voice sessions cost
more than text turns, so the cost-exhaustion abuse surface is bigger).
Sliding-window counter.

Architecture gap closed: this used to be in-memory-only, single-process --
correct for one gateway replica, but counts silently reset on a per-instance
basis the moment there's more than one (a client's next request landing on a
different instance gets a fresh window, defeating the whole limit). Backed
by Redis (a sorted-set sliding-window-log, one ZSET per key) when REDIS_URL
is configured, sharing counts across every replica; falls back to the exact
previous in-memory behavior when it isn't -- zero behavior change for a
single-instance deployment, matching this codebase's own "optional at
runtime" convention (chat_store.py/POSTGRES_DSN, rag/embeddings.py/
HF_API_TOKEN) rather than a new required dependency.
"""

from __future__ import annotations

import os
import time
import uuid
from collections import defaultdict, deque

import redis.asyncio as redis


class SlidingWindowRateLimiter:
    def __init__(self, *, max_requests: int, window_seconds: float, redis_url_env: str = "REDIS_URL"):
        self._max_requests = max_requests
        self._window_seconds = window_seconds
        self._hits: dict[str, deque] = defaultdict(deque)  # in-memory fallback only
        self._redis_url_env = redis_url_env
        self._redis: redis.Redis | None = None
        self._redis_checked = False

    def _get_redis(self) -> "redis.Redis | None":
        # Lazy + memoized, same pattern as chat_store._get_pool() -- never
        # touches the network just from constructing this object, only from
        # the first real allow() call, and only once per process.
        if not self._redis_checked:
            self._redis_checked = True
            url = os.environ.get(self._redis_url_env)
            if url:
                self._redis = redis.from_url(url)
        return self._redis

    async def allow(self, key: str, *, now: float | None = None) -> bool:
        r = self._get_redis()
        if r is not None:
            return await self._allow_redis(r, key, now)
        return self._allow_in_memory(key, now)

    async def _allow_redis(self, r: "redis.Redis", key: str, now: float | None) -> bool:
        # Wall-clock time, not time.monotonic() -- monotonic clocks aren't
        # comparable across processes, and this window must be shared across
        # every replica for the limit to actually mean anything cross-process.
        now_score = now if now is not None else time.time()
        cutoff = now_score - self._window_seconds
        redis_key = f"ratelimit:{key}"
        await r.zremrangebyscore(redis_key, "-inf", cutoff)
        count = await r.zcard(redis_key)
        if count >= self._max_requests:
            return False
        # Unique member per hit (timestamp alone can collide within the same
        # millisecond under real load) -- a ZSET member must be unique or a
        # second hit at the same score overwrites the first instead of
        # adding a second entry, undercounting real concurrent requests.
        await r.zadd(redis_key, {f"{now_score}:{uuid.uuid4().hex}": now_score})
        await r.expire(redis_key, int(self._window_seconds) + 1)
        return True

    def _allow_in_memory(self, key: str, now: float | None) -> bool:
        now = now if now is not None else time.monotonic()
        hits = self._hits[key]
        cutoff = now - self._window_seconds
        while hits and hits[0] < cutoff:
            hits.popleft()
        # Real gap: a key whose deque drains to empty (no requests within the
        # window) stayed in the dict forever -- every distinct client key
        # ever seen was a permanent entry, unbounded memory growth on a
        # long-running process. Drop it here and only re-add on an actual
        # new hit below, so a key with no recent activity doesn't linger.
        if not hits:
            del self._hits[key]
        if len(hits) >= self._max_requests:
            return False
        hits.append(now)
        self._hits[key] = hits
        return True
