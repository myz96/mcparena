"""Tests for MCP tool wrapping — especially that the tool arg-schema reaches dspy.

Regression guard for the harness bug where `make_tools` exposed only `**kwargs`
to dspy's signature introspection, hiding each MCP tool's real parameter names
from the program LM (which then guessed wrong kwargs and every call failed with
"Missing required arguments"). The fix forwards `inputSchema` to `dspy.Tool`.
"""

from __future__ import annotations

from typing import Any

from mcparena.pilot import tools

_CONVERT_TIME_SPEC = {
    "name": "convert_time",
    "description": "Convert time between timezones",
    "inputSchema": {
        "type": "object",
        "properties": {
            "source_timezone": {"type": "string", "description": "Source IANA timezone name"},
            "time": {"type": "string", "description": "Time to convert in 24-hour format (HH:MM)"},
            "target_timezone": {"type": "string", "description": "Target IANA timezone name"},
        },
        "required": ["source_timezone", "time", "target_timezone"],
    },
}

_NO_SCHEMA_SPEC = {
    "name": "ping",
    "description": "No-arg health check",
    "inputSchema": {"type": "object", "properties": {}},
}


class _StubSession:
    """Minimal stand-in for PersistentMCPSession — only `tool_specs` is read by make_tools."""

    def __init__(self, specs: list[dict[str, Any]]) -> None:
        self._specs = specs

    @property
    def tool_specs(self) -> list[dict[str, Any]]:
        return self._specs

    def call_tool(
        self, tool_name: str, **kwargs: Any
    ) -> str:  # pragma: no cover - not invoked here
        return ""


def test_make_tools_forwards_inputschema_arg_names() -> None:
    [tool] = tools.make_tools(_StubSession([_CONVERT_TIME_SPEC]))  # type: ignore[arg-type]
    # The model must see the real arg names, not an empty {'kwargs': {}}.
    assert set(tool.args.keys()) == {"source_timezone", "time", "target_timezone"}


def test_make_tools_forwards_arg_descriptions() -> None:
    [tool] = tools.make_tools(_StubSession([_CONVERT_TIME_SPEC]))  # type: ignore[arg-type]
    rendered = str(tool)
    assert "HH:MM" in rendered  # the per-arg description reaches the model
    assert "Source IANA timezone name" in rendered


def test_make_tools_handles_tools_without_args() -> None:
    # A no-arg tool must still construct cleanly (no args forwarded).
    [tool] = tools.make_tools(_StubSession([_NO_SCHEMA_SPEC]))  # type: ignore[arg-type]
    assert tool.name == "ping"
    assert not tool.args


def test_make_tools_preserves_tool_name_and_desc() -> None:
    [tool] = tools.make_tools(_StubSession([_CONVERT_TIME_SPEC]))  # type: ignore[arg-type]
    assert tool.name == "convert_time"
    assert "Convert time between timezones" in (tool.desc or "")
