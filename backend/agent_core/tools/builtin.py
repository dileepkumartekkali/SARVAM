"""Real, dependency-free builtin tools — no API keys, no network, genuinely
executable. Enough to prove the agent loop does real work, not enough to be
a product tool catalog; add real integrations (search, calendar, orders)
behind this same `ToolSpec` shape when they exist.
"""

from __future__ import annotations

import ast
import operator
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .registry import ToolRegistry, ToolSpec

# --- get_current_datetime ---------------------------------------------------


async def get_current_datetime(timezone: str = "UTC") -> str:
    try:
        tz = ZoneInfo(timezone)
    except (ZoneInfoNotFoundError, ValueError):
        return f"Error: unknown timezone '{timezone}'. Use an IANA name like 'Asia/Kolkata'."
    now = datetime.now(tz)
    return now.strftime("%A, %d %B %Y, %H:%M:%S %Z")


# --- calculate ---------------------------------------------------------------

_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_ALLOWED_UNARYOPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _safe_eval(node: ast.AST) -> float:
    """Evaluates ONLY numeric arithmetic — no names, no calls, no attribute
    access, no subscripting. `eval()` is exactly the wrong tool for
    LLM-supplied input; this walks the AST and rejects anything that isn't a
    number or an allowed operator, so it can't execute arbitrary code."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        return _ALLOWED_BINOPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARYOPS:
        return _ALLOWED_UNARYOPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("expression contains a disallowed construct")


async def calculate(expression: str) -> str:
    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree.body)
    except (SyntaxError, ValueError, ZeroDivisionError, TypeError) as e:
        return f"Error: couldn't evaluate '{expression}' ({e})"
    return f"{result:g}"


# --- convert_units -----------------------------------------------------------

_LENGTH_TO_METERS = {"m": 1.0, "km": 1000.0, "cm": 0.01, "mi": 1609.344, "ft": 0.3048, "in": 0.0254}
_WEIGHT_TO_KG = {"kg": 1.0, "g": 0.001, "lb": 0.453592, "oz": 0.0283495}


async def convert_units(value: float, from_unit: str, to_unit: str) -> str:
    from_unit, to_unit = from_unit.lower(), to_unit.lower()
    if from_unit in _LENGTH_TO_METERS and to_unit in _LENGTH_TO_METERS:
        meters = value * _LENGTH_TO_METERS[from_unit]
        return f"{meters / _LENGTH_TO_METERS[to_unit]:g} {to_unit}"
    if from_unit in _WEIGHT_TO_KG and to_unit in _WEIGHT_TO_KG:
        kg = value * _WEIGHT_TO_KG[from_unit]
        return f"{kg / _WEIGHT_TO_KG[to_unit]:g} {to_unit}"
    if {from_unit, to_unit} <= {"c", "f", "k"}:
        celsius = {"c": value, "f": (value - 32) * 5 / 9, "k": value - 273.15}[from_unit]
        result = {"c": celsius, "f": celsius * 9 / 5 + 32, "k": celsius + 273.15}[to_unit]
        return f"{result:g} {to_unit}"
    return f"Error: can't convert between '{from_unit}' and '{to_unit}' (unsupported or mismatched unit types)"


# --- save_note / delete_note (write-scope demo) ------------------------------
# In-memory, per-process — a real demonstration of a stateful, irreversible
# tool paired with security/confirmation.py's gate, not a production notes
# service. Swap for a real persistence layer without touching the gate logic.

_notes: dict[str, str] = {}
_next_note_id = 1


async def save_note(text: str) -> str:
    global _next_note_id
    note_id = str(_next_note_id)
    _next_note_id += 1
    _notes[note_id] = text
    return f"Saved as note #{note_id}."


async def delete_note(note_id: str) -> str:
    if note_id not in _notes:
        return f"Error: no note #{note_id}."
    del _notes[note_id]
    return f"Deleted note #{note_id}."


def build_default_registry() -> ToolRegistry:
    return ToolRegistry(
        [
            ToolSpec(
                name="get_current_datetime",
                description="Returns the current date and time.",
                parameters={
                    "timezone": {
                        "type": "string",
                        "description": "IANA timezone name, e.g. 'Asia/Kolkata'. Optional, defaults to UTC.",
                    }
                },
                fn=get_current_datetime,
            ),
            ToolSpec(
                name="calculate",
                description="Evaluates a numeric arithmetic expression (+ - * / // % **, parentheses).",
                parameters={"expression": {"type": "string", "description": "e.g. '12 * (3 + 4)'"}},
                required=["expression"],
                fn=calculate,
            ),
            ToolSpec(
                name="convert_units",
                description="Converts a value between compatible units (length, weight, or temperature).",
                parameters={
                    "value": {"type": "number", "description": "the numeric value to convert"},
                    "from_unit": {"type": "string", "description": "e.g. 'km', 'lb', 'c'"},
                    "to_unit": {"type": "string", "description": "e.g. 'mi', 'kg', 'f'"},
                },
                required=["value", "from_unit", "to_unit"],
                fn=convert_units,
            ),
            ToolSpec(
                name="save_note",
                description="Saves a short text note for the user and returns its id.",
                parameters={"text": {"type": "string", "description": "the note content"}},
                required=["text"],
                fn=save_note,
            ),
            ToolSpec(
                name="delete_note",
                description="Permanently deletes a saved note by id.",
                parameters={"note_id": {"type": "string", "description": "the note id returned by save_note"}},
                required=["note_id"],
                fn=delete_note,
                write_scope=True,
            ),
        ]
    )
