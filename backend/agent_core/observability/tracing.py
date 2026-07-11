"""Distributed tracing across gateway -> graph -> LLM -> TTS.

Real OpenTelemetry spans, not a hand-rolled trace-id scheme. Default exporter
is `ConsoleSpanExporter` — spans print as structured output with zero
collector infrastructure, so this is genuinely exercised in dev/CI, not
theoretical. Set `OTEL_EXPORTER_OTLP_ENDPOINT` to send to a real backend
(Jaeger/Tempo/etc.) via the standard OTLP exporter instead — swapping the
exporter is the only change; every `start_span()` call site is unaffected.
"""

from __future__ import annotations

import os
from contextlib import contextmanager

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter, SimpleSpanProcessor

_initialized = False


def init_tracing(service_name: str) -> None:
    """Idempotent — safe to call at import time in each service's entrypoint."""
    global _initialized
    if _initialized:
        return
    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))

    otlp_endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if otlp_endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        except ImportError as e:
            raise RuntimeError(
                "OTEL_EXPORTER_OTLP_ENDPOINT is set but "
                "opentelemetry-exporter-otlp-proto-http isn't installed — "
                "pip install it to send traces to a real collector."
            ) from e
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=otlp_endpoint)))
    else:
        # SimpleSpanProcessor, not Batch: a console write is synchronous and
        # cheap — no background export thread to race process shutdown
        # (e.g. in short-lived test runs) for zero batching benefit.
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

    trace.set_tracer_provider(provider)
    _initialized = True


def get_tracer(name: str):
    return trace.get_tracer(name)


@contextmanager
def start_span(tracer_name: str, span_name: str, **attributes):
    """`with start_span("speech_gateway", "stt.stream", session_id=...): ...`
    — a span's parent is whatever span is active in the current context
    (contextvars-based), so gateway -> graph -> LLM -> TTS spans nest
    correctly across the async call chain without threading a context object
    through every function signature by hand.
    """
    tracer = get_tracer(tracer_name)
    with tracer.start_as_current_span(span_name) as span:
        for key, value in attributes.items():
            span.set_attribute(key, value)
        yield span


def current_trace_id() -> str | None:
    """Hex trace id of the active span, for correlating log lines to traces."""
    span = trace.get_current_span()
    ctx = span.get_span_context()
    if ctx is None or ctx.trace_id == 0:
        return None
    return format(ctx.trace_id, "032x")
