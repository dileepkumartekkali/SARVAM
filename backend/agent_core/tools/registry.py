"""Real tool registry — the missing piece that turns task_agent's reasoning
loop from agentic *scaffolding* into an actual agent: a place where tools are
registered with a name, a description, and a real JSON Schema, all of which
feed both the native provider function-calling path (`as_tool_definitions()`)
and the legacy prompted-text fallback (`as_prompt_manifest()`) for providers/
models that don't respect the native `tools` field.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Awaitable, Callable

from ..llm_adapter.base import ToolDefinition

ToolFn = Callable[..., Awaitable[str]]


@dataclass
class ToolSpec:
    name: str
    description: str
    # JSON Schema "properties" object, e.g. {"expression": {"type": "string",
    # "description": "e.g. '12 * (3 + 4)'"}} — real schema, not a
    # human-readable string, so it can be sent as-is to a provider's native
    # function-calling API (see as_tool_definitions()).
    parameters: dict
    fn: ToolFn
    required: list[str] = field(default_factory=list)
    write_scope: bool = False  # irreversible — see security/confirmation.py


class ToolRegistry:
    def __init__(self, specs: list[ToolSpec] | None = None):
        self._specs: dict[str, ToolSpec] = {s.name: s for s in (specs or [])}

    def register(self, spec: ToolSpec) -> None:
        self._specs[spec.name] = spec

    def as_dispatch_dict(self) -> dict[str, ToolFn]:
        return {name: spec.fn for name, spec in self._specs.items()}

    def write_scope_names(self) -> set[str]:
        return {name for name, spec in self._specs.items() if spec.write_scope}

    def as_tool_definitions(self) -> list[ToolDefinition]:
        """The native function-calling schema — each provider adapter
        translates these into its own wire format (OpenAI `tools`, Anthropic
        `input_schema`, Gemini `function_declarations`)."""
        return [
            ToolDefinition(name=s.name, description=s.description, parameters=s.parameters, required=s.required)
            for s in self._specs.values()
        ]

    def as_prompt_manifest(self) -> str:
        """Legacy prompted-text fallback: what tools exist, their params, and
        the `TOOL_CALL: {...}` syntax — for a provider/model that doesn't
        respect the native `tools` field. task_agent tries native tool calls
        first and only falls back to parsing this convention out of the
        model's plain text if none came back natively."""
        if not self._specs:
            return ""
        lines = [
            "## TOOLS",
            "You have access to the following tools. Prefer using them via your",
            "native tool-calling mechanism. If that isn't available, call one by",
            'outputting a line of the exact form: TOOL_CALL: {"name": "<tool_name>", '
            '"args": {<argument object>}}',
            "That line must be the ENTIRE reply — no lead-in like \"I'll search "
            "for that\" or \"Let me check\" before it. Any text before TOOL_CALL "
            "gets shown/spoken to the user before the tool result exists.",
            "Only call a tool when you genuinely need it — never fabricate a",
            "result a tool could have given you, and never claim to have called",
            "a tool you didn't.",
            "",
        ]
        for spec in self._specs.values():
            param_descriptions = ", ".join(
                f"{k} ({v.get('type', 'any')}{' — ' + v['description'] if v.get('description') else ''})"
                for k, v in spec.parameters.items()
            ) or "no arguments"
            scope_note = " — IRREVERSIBLE, requires user confirmation" if spec.write_scope else ""
            lines.append(f"- {spec.name}({param_descriptions}): {spec.description}{scope_note}")
        return "\n".join(lines)
