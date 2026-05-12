"""MCP-Bench loader contract.

Hermetic tests for `parse_server_tasks`: signature, constants, the
empty-result fallback when the pinned MCP-Bench clone is absent, the
typo-raises-KeyError path when the clone exists but the id is missing,
and the populated path with a synthetic fixture mirroring MCP-Bench's
actual JSON shape.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mcparena.pilot.benchmark import (
    DEFAULT_DEST,
    MCP_BENCH_REPO,
    SINGLE_TASKS_FILE,
    parse_server_tasks,
)


def test_constants_set() -> None:
    assert MCP_BENCH_REPO == "https://github.com/Accenture/mcp-bench"
    assert DEFAULT_DEST == Path("third_party/mcp-bench-tasks")


def test_parse_returns_empty_when_source_missing(tmp_path: Path) -> None:
    """When the MCP-Bench clone is absent, the loader returns [] (not raise)."""
    # tmp_path has no SINGLE_TASKS_FILE — simulates not-yet-cloned
    result = parse_server_tasks("Math MCP", source=tmp_path)
    assert result == []


def _write_fixture(tmp_path: Path, server_name: str) -> Path:
    """Build a minimal MCP-Bench-shaped tasks JSON fixture for one server."""
    fixture_path = tmp_path / SINGLE_TASKS_FILE
    fixture_path.parent.mkdir(parents=True, exist_ok=True)
    fixture_path.write_text(
        json.dumps(
            {
                "server_tasks": [
                    {
                        "server_name": server_name,
                        "combination_type": "single",
                        "tasks": [
                            {
                                "task_id": f"{server_name.lower().replace(' ', '_')}_000",
                                "task_description": "compute the mean of [1, 2, 3]",
                                "fuzzy_description": "average a tiny array",
                                "dependency_analysis": "uses Math MCP mean tool",
                                "distraction_servers": ["Unit Converter"],
                            }
                        ],
                    }
                ]
            }
        )
    )
    return fixture_path


def test_parse_returns_dspy_examples_with_field_mapping(tmp_path: Path) -> None:
    """When source IS cloned and id matches, returns dspy.Example list with the
    documented field mapping (task_description -> user_request)."""
    _write_fixture(tmp_path, "Math MCP")
    result = parse_server_tasks("Math MCP", source=tmp_path)
    assert len(result) == 1
    ex = result[0]
    assert ex.task_id == "math_mcp_000"
    assert ex.user_request == "compute the mean of [1, 2, 3]"
    assert ex.mcp_bench_fuzzy == "average a tiny array"
    # `.with_inputs("user_request")` was called — input_keys reflects this
    assert "user_request" in ex.inputs().toDict()


def test_parse_raises_keyerror_on_typo_when_source_present(tmp_path: Path) -> None:
    """If the clone exists but the id is wrong, raise KeyError (not silently
    return [] — that would mask typos and let pilots run with 0 trials)."""
    _write_fixture(tmp_path, "Math MCP")
    with pytest.raises(KeyError, match="Math Mcp"):  # case-sensitive
        parse_server_tasks("Math Mcp", source=tmp_path)
