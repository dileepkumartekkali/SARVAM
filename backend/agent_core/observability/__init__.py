"""Observability: structured logging, distributed tracing, and metrics.

Three pieces, each a thin wrapper over a standard tool rather than a
hand-rolled protocol:

  logging_config  stdlib `logging` + a JSON formatter, correlated with the
                  active trace/span IDs (agent_system_prompt.md §4: log the
                  prompt version + turn_trace, never the full prompt text)
  tracing         OpenTelemetry SDK — spans propagate gateway -> graph -> LLM
                  -> TTS via contextvars; ConsoleSpanExporter by default (real
                  output, zero collector infra), OTLP exporter opt-in via env
  metrics         prometheus_client — per-stage latency histograms (STT, LLM
                  TTFB, TTS TTFB) and counters (errors, reconnects), exposed
                  at /metrics on both FastAPI apps
"""
