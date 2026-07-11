# Load Test Report

**Actually run**, against real (running) `backend` and `speech-gateway`
processes on a single Windows dev machine, using
[`scripts/load_test.py`](../backend/scripts/load_test.py). Both apps ran in
`MAAV_LOAD_TEST_MODE=true`, which swaps the real Sarvam/LLM clients for
minimal-latency fakes (`agent_core/speech_gateway/load_test_fakes.py`,
`agent_core/api/load_test_fakes.py`) — there is no way to responsibly load
test 10,000 concurrent sessions against a real, billed, rate-limited
third-party API without an account explicitly provisioned for that, which
doesn't exist in this environment. **What this measures**: our own
gateway/backend code's concurrency-handling capacity. **What it does not
measure**: Sarvam's real rate ceiling — see "Unknown: Sarvam's real ceiling"
below.

**Important caveat**: load generator and both servers ran on the same
machine, sharing one CPU/network stack. Absolute latency numbers are not
representative of a real multi-node production deployment (the generator
itself competes for CPU with the server under test). The bottleneck
*comparison between services* and the *bug found* are real and transferable;
the absolute millisecond figures are not.

## Results

| Tier | Path | Concurrency cap | Wall time | Success | Error rate | p50 | p95 | p99 |
|---|---|---|---|---|---|---|---|---|
| 100 | STT WS | 500 | 2.35s | 100/100 | 0.0% | 2290ms | 2306ms | 2310ms |
| 1,000 | STT WS | 1,000 | 3.62s | 1000/1000 | 0.0% | 3155ms | 3372ms | 3383ms |
| 10,000 | STT WS | 2,000 | 20.36s | 9,517/10,000 | 4.8% | 10,805ms | 18,932ms | 19,687ms |
| 100 | `/chat` | 500 | 0.83s | 100/100 | 0.0% | 570ms | 576ms | 577ms |
| 1,000 | `/chat` | 500 | 17.73s | 1,000/1,000 | 0.0% | 11,910ms | 15,451ms | 15,539ms |

(10,000-tier `/chat` was not run — at the 1,000-tier's already-heavy
per-request cost, extrapolating to 10,000 would mean tens of minutes against
a single dev-machine process; the 1,000-tier result was sufficient to
identify the bottleneck class, see below.)

## What actually broke first, with data

**Not** what was assumed going in. The three candidates named in the task
were Speech Gateway connection count, backend CPU, and Sarvam's own rate
ceiling. Measured:

1. **A real concurrency bug, found by this test, not assumed** — at the
   10,000-connection STT tier, the *first* run hit a **47.9% error rate**
   (4,931/10,000 failed), all `RuntimeError: Unexpected ASGI message
   'websocket.close', after sending 'websocket.close' or response already
   completed`. This is a race in `agent_core/speech_gateway/main.py`: under
   normal load, a client disconnect and the route's own `finally:
   await websocket.close()` never overlap; at thousands of concurrent
   connects/disconnects, they do, and uvicorn raises on the second close.
   **Fixed** (`_safe_close()`, swallows the redundant-close `RuntimeError`)
   and reran the identical 10,000-tier test: **error rate dropped to 4.8%**
   (483/10,000). This is the single highest-value finding in this report —
   a load test the codebase had never actually been subjected to caught a
   real bug the unit/integration test suite's low concurrency never
   exercised.

2. **After the fix, the STT WS path degrades gracefully, not catastrophically**
   — at 10,000 connections (2,000 concurrent), p50 latency rose to ~10.8s
   and the remaining ~4.8% failures showed **no server-side errors** in the
   gateway log at all; they're consistent with the load generator's own 10s
   client-side timeout being exceeded under queueing delay, not a server
   rejection. This points to **single-process asyncio/event-loop scheduling
   throughput** as the ceiling for this service at this concurrency, on this
   one machine — not a hard connection-count wall, and not an error the
   server itself raised.

3. **The backend's `/chat` path is far more CPU-expensive per request than
   the gateway's WS handling**, at equal concurrency: 1,000 STT connections
   returned in 3.6s wall time; 1,000 `/chat` requests took **17.7s** wall
   time with a **~12s p50** — roughly 4-5x worse — despite the fake LLM call
   itself taking only 20ms. The difference is the LangGraph orchestration
   per request (language detection, pydantic validation, JSON structured
   logging, tracing spans) — real CPU-bound work that scales with request
   volume regardless of how fast the LLM answers. **This confirms backend
   CPU, not connection count, is this service's bottleneck** — matching the
   HPA config already written in `infra/k8s/backend-deployment-green.yaml`
   (scales on CPU utilization) rather than connection count (which is what
   the Speech Gateway's HPA scales on instead, correctly, per finding #2).

4. **The rate limiter is the actual *first* thing hit in any naive
   multi-connection scenario from one IP** — `STT_SESSION_RATE_LIMIT`
   defaults to 10 requests/60s per client IP (`security/rate_limit.py`).
   This load test only produced real capacity data because the limit was
   raised via env var for the test run. In production, any legitimate
   traffic source behind a shared IP (a corporate NAT, a mobile carrier
   NAT) would hit this ceiling **long before** the connection-count or CPU
   ceilings above become relevant. This is flagged in
   `infra/observability/alerts.yml`'s `RateLimiterSaturated` alert — but the
   underlying design (per-IP, not per-authenticated-user) is worth
   reconsidering; per-JWT-subject keying would avoid penalizing legitimate
   shared-IP traffic while still stopping a single abusive account.

## Unknown: Sarvam's real ceiling

Not measured, and not measurable in this environment: Sarvam's own
account-level rate limits and concurrent-connection ceiling are not public
in the docs consulted during Phase 4, and no live account exists here to
probe them empirically. **This must be confirmed directly with Sarvam
(account manager or their docs' current rate-limit section) before setting
the Speech Gateway's autoscaling target in production** — the HPA's
`averageValue: "400"` connections/pod in `speech-gateway-rollout.yaml` is
derived from this session's single-machine capacity test, not from Sarvam's
actual ceiling, and is explicitly a placeholder pending that confirmation.

## Bottom line

At the scale this session could actually test (single machine, fakes
standing in for Sarvam): the Speech Gateway's own code was the first thing
to break — via a real bug, not a capacity wall — and after fixing it,
degrades gracefully rather than falling over. The backend is meaningfully
more CPU-hungry per request than the gateway's connection handling, which is
the right basis for the two services' differently-shaped autoscaling
configs already written in `infra/k8s/`. The one bottleneck that could not be
measured at all — Sarvam's real ceiling — is the one most likely to matter
in an actual production deployment, and needs to be confirmed out-of-band.
