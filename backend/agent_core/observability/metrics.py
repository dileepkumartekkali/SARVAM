"""Prometheus metrics: per-stage latency (STT, LLM TTFB, TTS TTFB) and the
counters alerting.yml's rules fire on (errors, reconnects).

`prometheus_client`'s default registry is process-global by design — each of
the two services (backend, speech gateway) is its own process, so there's no
cross-contamination between their /metrics endpoints.
"""

from __future__ import annotations

import time
from contextlib import contextmanager

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

# Buckets tuned for voice-latency scales: sub-100ms (STT partials, LLM TTFB
# target) through multi-second (slow LLM fallback chains, TTS on a cold
# socket) — not the default 5ms-10s web-request buckets.
_LATENCY_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30)

stt_latency_seconds = Histogram(
    "maav_stt_latency_seconds", "STT stage latency (VAD speech_end -> final transcript)", buckets=_LATENCY_BUCKETS
)
llm_ttfb_seconds = Histogram(
    "maav_llm_ttfb_seconds", "LLM time-to-first-byte (request sent -> first token)", buckets=_LATENCY_BUCKETS
)
llm_latency_seconds = Histogram(
    "maav_llm_latency_seconds", "Non-streaming LLM call latency (request -> full response)", buckets=_LATENCY_BUCKETS
)
tts_ttfb_seconds = Histogram(
    "maav_tts_ttfb_seconds", "TTS time-to-first-byte (text sent -> first audio chunk)", buckets=_LATENCY_BUCKETS
)

errors_total = Counter("maav_errors_total", "Errors by stage", ["stage"])
reconnects_total = Counter("maav_reconnects_total", "Client reconnect attempts by stage", ["stage"])
active_voice_sessions = Counter("maav_voice_sessions_started_total", "Voice sessions started")

# Gauge, not the counter above: autoscaling (speech-gateway-rollout.yaml)
# scales on concurrently-open connections, which only a gauge (inc on
# connect, dec on disconnect) can express — the counter only ever grows.
active_websocket_connections = Gauge(
    "maav_active_websocket_connections", "Currently-open STT/TTS WebSocket connections", ["kind"]
)


@contextmanager
def observe_latency(histogram: Histogram):
    start = time.monotonic()
    try:
        yield
    finally:
        histogram.observe(time.monotonic() - start)


def metrics_response() -> tuple[bytes, str]:
    """Returns (body, content_type) for a `/metrics` route handler."""
    return generate_latest(), CONTENT_TYPE_LATEST
