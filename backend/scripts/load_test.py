"""Load test: 100 -> 1,000 -> 10,000 concurrent voice sessions against the
Speech Gateway's /ws/stt (fake STT client — see load_test_fakes.py for why),
plus a comparison run against the backend's /chat (fake LLM provider).

Reports p50/p95/p99 latency and error rate per tier, so the "where does it
break first" question is answered with numbers, not a guess. Requires both
apps running with MAAV_LOAD_TEST_MODE=true (see docs/LOAD_TEST_REPORT.md for
exact invocation).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import time

import httpx
import jwt
import websockets

GATEWAY_WS_URL = "ws://localhost:8101/ws/stt"
BACKEND_URL = "http://localhost:8001"


def _valid_frame() -> bytes:
    expected_bytes = int(16000 * 0.032) * 2
    return b"\x00\x01" * (expected_bytes // 2)


async def _one_stt_session(sem: asyncio.Semaphore) -> tuple[bool, float]:
    start = time.monotonic()
    async with sem:
        try:
            async with websockets.connect(GATEWAY_WS_URL, open_timeout=10) as ws:
                await ws.send(json.dumps({"codec": "pcm_s16le", "sample_rate": 16000}))
                for _ in range(3):
                    await ws.send(_valid_frame())
                await asyncio.wait_for(ws.recv(), timeout=10)
            return True, time.monotonic() - start
        except Exception:
            return False, time.monotonic() - start


async def _one_chat_request(client: httpx.AsyncClient, token: str, sem: asyncio.Semaphore, i: int) -> tuple[bool, float]:
    start = time.monotonic()
    async with sem:
        try:
            resp = await client.post(
                f"{BACKEND_URL}/chat",
                json={
                    "session_id": f"load-{i}",
                    "conversation_id": f"load-{i}",
                    "thread_id": f"load-{i}",
                    "message": "hello, quick load test message",
                },
                headers={"Authorization": f"Bearer {token}"},
                timeout=15,
            )
            return resp.status_code == 200, time.monotonic() - start
        except Exception:
            return False, time.monotonic() - start


def _report(tier: int, kind: str, results: list[tuple[bool, float]], wall_seconds: float) -> None:
    successes = [r for ok, r in results if ok]
    failures = len(results) - len(successes)
    latencies = sorted(successes)
    if latencies:
        p50 = statistics.median(latencies)
        p95 = latencies[int(len(latencies) * 0.95) - 1] if len(latencies) > 1 else latencies[0]
        p99 = latencies[int(len(latencies) * 0.99) - 1] if len(latencies) > 1 else latencies[0]
    else:
        p50 = p95 = p99 = float("nan")
    print(
        f"[{kind}] tier={tier:>6} wall={wall_seconds:6.2f}s ok={len(successes):>6} fail={failures:>6} "
        f"error_rate={failures / len(results):5.1%} p50={p50*1000:7.1f}ms p95={p95*1000:7.1f}ms p99={p99*1000:7.1f}ms"
    )


async def run_stt_tier(tier: int, concurrency_cap: int) -> None:
    sem = asyncio.Semaphore(concurrency_cap)
    start = time.monotonic()
    results = await asyncio.gather(*[_one_stt_session(sem) for _ in range(tier)])
    _report(tier, "stt_ws", results, time.monotonic() - start)


async def run_chat_tier(tier: int, token: str, concurrency_cap: int) -> None:
    sem = asyncio.Semaphore(concurrency_cap)
    start = time.monotonic()
    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(*[_one_chat_request(client, token, sem, i) for i in range(tier)])
    _report(tier, "chat_http", results, time.monotonic() - start)


def get_dev_token() -> str:
    # No server-side token-issuing route exists anymore (real auth is
    # Supabase-issued JWTs) — mint one locally with the same shared secret
    # the backend verifies against, exactly like the old /auth/dev-login did.
    secret = os.environ.get("JWT_SIGNING_SECRET", "dev-preview-secret-not-for-prod")
    return jwt.encode({"sub": "load-test", "exp": int(time.time()) + 3600}, secret, algorithm="HS256")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiers", type=int, nargs="+", default=[100, 1000, 10000])
    parser.add_argument("--concurrency-cap", type=int, default=500, help="max in-flight connections at once")
    parser.add_argument("--skip-chat", action="store_true")
    args = parser.parse_args()

    for tier in args.tiers:
        await run_stt_tier(tier, args.concurrency_cap)

    if not args.skip_chat:
        token = get_dev_token()
        for tier in args.tiers:
            await run_chat_tier(tier, token, args.concurrency_cap)


if __name__ == "__main__":
    asyncio.run(main())
