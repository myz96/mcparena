"""MCP-Bench loader stub contract.

Unit 3 ships `parse_server_tasks` as a stub returning an empty list. The real
parser lands during Unit 5's Day-1 sub-sequence once the MCP-Bench task JSON
format is inspected on the pinned commit. This test verifies the stub contract
so anything that breaks the function signature is caught early.
"""

from pathlib import Path

from mcparena.pilot.benchmark import DEFAULT_DEST, MCP_BENCH_REPO, parse_server_tasks


def test_constants_set() -> None:
    assert MCP_BENCH_REPO == "https://github.com/Accenture/mcp-bench"
    assert DEFAULT_DEST == Path("third_party/mcp-bench-tasks")


def test_parse_returns_list_for_unknown_server() -> None:
    """Stub returns [] for any server id (real impl populates from JSON)."""
    result = parse_server_tasks("nonexistent-server")
    assert isinstance(result, list)
    assert result == []


def test_parse_accepts_custom_source_path(tmp_path: Path) -> None:
    """Loader accepts a custom source dir (for test isolation)."""
    result = parse_server_tasks("filesystem", source=tmp_path)
    assert isinstance(result, list)
