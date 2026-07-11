import json
import logging

from agent_core.observability.logging_config import JsonFormatter
from agent_core.observability.tracing import current_trace_id, get_tracer, init_tracing


def test_json_formatter_includes_extra_fields_and_no_full_prompt():
    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="agent_core.turn_trace",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="turn_trace",
        args=(),
        exc_info=None,
    )
    record.prompt_version = "text_mode_system.v1"
    record.tool_call_count = 2

    parsed = json.loads(formatter.format(record))

    assert parsed["message"] == "turn_trace"
    assert parsed["prompt_version"] == "text_mode_system.v1"
    assert parsed["tool_call_count"] == 2
    assert "prompt_text" not in parsed  # never the full prompt — only the version id


def test_trace_id_present_inside_a_span():
    init_tracing("test-service")
    tracer = get_tracer("test")
    assert current_trace_id() is None  # no active span outside one
    with tracer.start_as_current_span("unit-test-span"):
        trace_id = current_trace_id()
        assert trace_id is not None
        assert len(trace_id) == 32  # 128-bit hex trace id
