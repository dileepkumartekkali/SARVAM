"""Structured (JSON) logging, correlated with the active trace id.

Per agent_system_prompt.md §4: log the prompt *version* and turn_trace
fields, never the full prompt text — `task_agent.py`'s existing
`logger.info("turn_trace", extra={...})` calls are unchanged; this module
only changes how those records are *formatted* (JSON, one line, trace-
correlated) so a log aggregator can index and query them.
"""

from __future__ import annotations

import json
import logging
import sys

from .tracing import current_trace_id

_RESERVED = frozenset(logging.LogRecord(None, 0, "", 0, "", (), None).__dict__.keys()) | {"message"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        trace_id = current_trace_id()
        if trace_id:
            payload["trace_id"] = trace_id
        # Everything passed via `extra={...}` (e.g. prompt_version,
        # tool_call_count, self_check_ok) rides along as-is.
        for key, value in record.__dict__.items():
            if key not in _RESERVED:
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: int = logging.INFO) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)
