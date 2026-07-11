from agent_core.security.rate_limit import SlidingWindowRateLimiter


def test_allows_up_to_the_limit_then_blocks():
    limiter = SlidingWindowRateLimiter(max_requests=3, window_seconds=60)
    key = "user-1"

    assert limiter.allow(key, now=0) is True
    assert limiter.allow(key, now=1) is True
    assert limiter.allow(key, now=2) is True
    assert limiter.allow(key, now=3) is False


def test_window_slides_and_old_hits_expire():
    limiter = SlidingWindowRateLimiter(max_requests=2, window_seconds=10)
    key = "user-1"

    assert limiter.allow(key, now=0) is True
    assert limiter.allow(key, now=1) is True
    assert limiter.allow(key, now=2) is False  # over limit within the window

    assert limiter.allow(key, now=11) is True  # first hit (t=0) has aged out


def test_keys_are_independent():
    limiter = SlidingWindowRateLimiter(max_requests=1, window_seconds=60)

    assert limiter.allow("ip-a", now=0) is True
    assert limiter.allow("ip-b", now=0) is True
    assert limiter.allow("ip-a", now=1) is False
