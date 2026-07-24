from agent_core.security.rate_limit import SlidingWindowRateLimiter


class FakeRedis:
    """Minimal in-memory stand-in for the 4 redis.asyncio.Redis calls
    _allow_redis actually makes -- no real Redis needed to test that path."""

    def __init__(self):
        self._zsets: dict[str, dict[str, float]] = {}

    async def zremrangebyscore(self, key, min_score, max_score):
        z = self._zsets.get(key, {})
        cutoff = float(max_score)
        for member in [m for m, score in z.items() if score <= cutoff]:
            del z[member]

    async def zcard(self, key):
        return len(self._zsets.get(key, {}))

    async def zadd(self, key, mapping):
        self._zsets.setdefault(key, {}).update(mapping)

    async def expire(self, key, seconds):
        pass  # no TTL simulation needed for these tests


async def test_allows_up_to_the_limit_then_blocks():
    limiter = SlidingWindowRateLimiter(max_requests=3, window_seconds=60)
    key = "user-1"

    assert await limiter.allow(key, now=0) is True
    assert await limiter.allow(key, now=1) is True
    assert await limiter.allow(key, now=2) is True
    assert await limiter.allow(key, now=3) is False


async def test_window_slides_and_old_hits_expire():
    limiter = SlidingWindowRateLimiter(max_requests=2, window_seconds=10)
    key = "user-1"

    assert await limiter.allow(key, now=0) is True
    assert await limiter.allow(key, now=1) is True
    assert await limiter.allow(key, now=2) is False  # over limit within the window

    assert await limiter.allow(key, now=11) is True  # first hit (t=0) has aged out


async def test_keys_are_independent():
    limiter = SlidingWindowRateLimiter(max_requests=1, window_seconds=60)

    assert await limiter.allow("ip-a", now=0) is True
    assert await limiter.allow("ip-b", now=0) is True
    assert await limiter.allow("ip-a", now=1) is False


async def test_redis_backed_path_enforces_the_same_sliding_window():
    """Architecture gap closed: counts used to be in-memory/single-process
    only -- with REDIS_URL configured, the same sliding-window semantics are
    now enforced through Redis instead, shareable across replicas."""
    limiter = SlidingWindowRateLimiter(max_requests=2, window_seconds=10)
    limiter._redis = FakeRedis()
    limiter._redis_checked = True  # skip the real REDIS_URL/from_url lookup

    assert await limiter.allow("user-1", now=0) is True
    assert await limiter.allow("user-1", now=1) is True
    assert await limiter.allow("user-1", now=2) is False  # over limit within the window
    assert await limiter.allow("user-1", now=11) is True  # first hit (t=0) has aged out
    assert await limiter.allow("ip-b", now=0) is True  # independent key, unaffected
