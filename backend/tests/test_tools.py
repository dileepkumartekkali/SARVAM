"""Real tool execution — no mocks. These tools do actual work."""

import pytest

from agent_core.tools import build_default_registry
from agent_core.tools.builtin import calculate, convert_units, delete_note, get_current_datetime, save_note
from agent_core.tools.registry import ToolRegistry, ToolSpec


async def test_calculate_evaluates_real_arithmetic():
    assert await calculate("2 + 2") == "4"
    assert await calculate("12 * (3 + 4)") == "84"
    assert await calculate("2 ** 10") == "1024"
    assert await calculate("10 / 4") == "2.5"


async def test_calculate_rejects_non_arithmetic():
    result = await calculate("__import__('os').system('echo pwned')")
    assert result.startswith("Error")


async def test_calculate_rejects_names_and_calls():
    result = await calculate("open('/etc/passwd').read()")
    assert result.startswith("Error")


async def test_calculate_handles_division_by_zero():
    result = await calculate("1 / 0")
    assert result.startswith("Error")


async def test_get_current_datetime_returns_a_real_formatted_timestamp():
    result = await get_current_datetime("Asia/Kolkata")
    assert "202" in result or "203" in result  # a real year renders somewhere in the string


async def test_get_current_datetime_rejects_unknown_timezone():
    result = await get_current_datetime("Not/A_Real_Zone")
    assert result.startswith("Error")


async def test_convert_units_length():
    assert await convert_units(1, "km", "m") == "1000 m"


async def test_convert_units_temperature():
    assert await convert_units(0, "c", "f") == "32 f"


async def test_convert_units_rejects_mismatched_types():
    result = await convert_units(1, "kg", "km")
    assert result.startswith("Error")


async def test_save_and_delete_note_round_trip():
    save_result = await save_note("buy milk")
    assert "note #" in save_result
    note_id = save_result.split("#")[1].rstrip(".")

    delete_result = await delete_note(note_id)
    assert f"Deleted note #{note_id}" in delete_result

    # Second delete of the same id fails — it's really gone.
    second = await delete_note(note_id)
    assert second.startswith("Error")


def test_default_registry_has_expected_tools_and_write_scope():
    registry = build_default_registry()
    dispatch = registry.as_dispatch_dict()
    assert set(dispatch) == {"get_current_datetime", "calculate", "convert_units", "save_note", "delete_note"}
    assert registry.write_scope_names() == {"delete_note"}


def test_registry_manifest_includes_syntax_and_tool_names():
    registry = build_default_registry()
    manifest = registry.as_prompt_manifest()
    assert "TOOL_CALL:" in manifest
    assert "calculate(" in manifest
    assert "IRREVERSIBLE" in manifest  # delete_note's write-scope note


def test_empty_registry_produces_no_manifest():
    assert ToolRegistry().as_prompt_manifest() == ""


async def test_custom_tool_spec_dispatches_correctly():
    async def echo(text: str) -> str:
        return f"echo: {text}"

    registry = ToolRegistry([ToolSpec(name="echo", description="Echoes text.", parameters={"text": "string"}, fn=echo)])
    result = await registry.as_dispatch_dict()["echo"](text="hi")
    assert result == "echo: hi"
