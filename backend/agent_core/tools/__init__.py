"""Permission-gated external actions (tools) — real implementations.

`ToolRegistry`/`ToolSpec` (registry.py) hold name + description + params +
the callable; `build_default_registry()` (builtin.py) wires a handful of
real, dependency-free tools (datetime, calculator, unit conversion, a
save/delete-note pair demonstrating the write-scope confirmation gate with
an actual stateful action, not just a test fake).
"""

from .builtin import build_default_registry
from .registry import ToolFn, ToolRegistry, ToolSpec

__all__ = ["ToolRegistry", "ToolSpec", "ToolFn", "build_default_registry"]
