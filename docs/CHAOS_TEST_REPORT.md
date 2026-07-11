# Chaos Test Report

**Actually run**, against real running processes, using a real process kill
(`Popen.kill()` → `TerminateProcess` on Windows, the SIGKILL-equivalent) —
not a simulated exception or an in-process cancellation token. This is a
materially different (and stronger) test than Phase 5's, which only verified
`DuplexSession`'s cancellation logic against fake `asyncio` tasks that were
never really running an external process. Script:
[`scripts/chaos_test.py`](../backend/scripts/chaos_test.py).

## Setup

- Backend (`agent_core.api.main:app`) started once, on `MAAV_LOAD_TEST_MODE`
  (fake LLM — no real provider needed) and kept running for the entire test.
- Speech Gateway started, health-checked, then killed and restarted by the
  script itself mid-run.
- **Checkpointer**: `MemorySaver` (the no-DSN-configured default) — this
  test kills the *gateway*, not the *backend*, so the checkpointer process
  never dies here. Testing the Postgres-backed checkpointer specifically
  (surviving a *backend* restart) needs a running Postgres instance, which
  isn't available in this sandbox (no Docker — same constraint noted since
  Phase 0). That's a materially different, stronger claim than what was
  actually tested here — see "What this does NOT prove" below.

## Results

| Part | What was tested | Result |
|---|---|---|
| 1 | Open a real STT WebSocket session, `kill()` the gateway process mid-session, confirm the connected client observes a disconnect within 5s rather than hanging forever | **PASS** — `ConnectionClosed`/timeout raised, not a hang |
| 2 | With the gateway process **completely dead** (confirmed: zero gateway process running), send two `/chat` turns on the same `thread_id` to the backend and confirm both succeed normally | **PASS** — both turns returned normal responses; the backend never touched the gateway process to serve `/chat` at all |
| 3 | Restart the gateway, open a fresh STT session, confirm it functions normally (not left in some broken state from the kill) | **PASS** — a real transcript event returned over a brand-new connection |

## Why Part 2 is the actual "checkpointer resume" evidence

The task's premise — "confirm checkpointer resume actually works under real
network chaos" — is true here for a specific, honest reason: **the LangGraph
checkpointer that holds conversation/turn state lives in the backend
process, not the Speech Gateway.** The gateway is a stateless STT/TTS proxy
(Phase 4 design) with no conversation state of its own (`DuplexSession`,
Phase 5, holds only transient in-process barge-in state for the duration of
one connection — nothing checkpointed). So "does the checkpointer survive a
gateway kill" was always going to be true by construction, *if* the
architecture is correctly stateless — Part 2 is the test that actually
verifies that architectural claim empirically instead of asserting it. Two
turns on the identical `thread_id`, served correctly, while zero gateway
process existed, is real evidence the statelessness boundary holds.

## What this does NOT prove

- **Backend process death.** This test never killed the backend — only the
  gateway. If the *backend* died mid-session with `MemorySaver` (in-memory,
  no persistence), all in-flight thread state would be lost — that's exactly
  why `langgraph-checkpoint-postgres` exists as the documented prod backend
  (`pyproject.toml`'s comment). That specific chaos scenario — kill the
  backend, restart it, confirm a Postgres-backed checkpointer resumes the
  same thread — needs a running Postgres instance and was not testable here
  (no Docker in this environment). This is a real, open gap, not a
  quietly-assumed pass.
- **Real network chaos** (packet loss, latency injection, partial reads) —
  this test used a clean process kill, which is a real and common failure
  mode (pod eviction, OOM-kill) but not the only chaos shape. Tools like
  `tc`/Toxiproxy would be the next layer, not attempted here.
- **Full-duplex barge-in continuity across a gateway kill** — Phase 5's
  `DuplexSession` state machine was not exercised by this test at all; it
  only exists within one gateway process's memory, so a gateway kill trivially
  destroys any in-flight barge-in state along with the connection. That's
  expected and consistent with the gateway being explicitly stateless by
  design — but it does mean a barge-in in progress at the exact moment of a
  gateway kill is lost, same as the WebSocket connection itself.
