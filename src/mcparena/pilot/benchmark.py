"""MCP-Bench (Accenture) task loader.

Responsibilities:
1. Clone Accenture/mcp-bench at a pinned commit into `third_party/mcp-bench-tasks/`
   (gitignored — cloned on demand, not vendored).
2. Parse the per-server task JSON into `dspy.Example` lists.

The actual data format and field names are inspected on first clone; helpers
here document the intended mapping per the plan's MCP-Bench -> dspy.Example
contract.

Stubbed in Unit 3; live integration runs during Unit 5's Day-1 sub-sequence.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

MCP_BENCH_REPO = "https://github.com/Accenture/mcp-bench"
MCP_BENCH_PINNED_REF = "main"  # update to a specific SHA once Day-1 review is done
DEFAULT_DEST = Path("third_party/mcp-bench-tasks")


def ensure_mcp_bench_cloned(dest: Path = DEFAULT_DEST) -> Path:
    """Clone or update Accenture/mcp-bench at the pinned ref. Idempotent."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        subprocess.run(["git", "clone", MCP_BENCH_REPO, str(dest)], check=True)
    subprocess.run(["git", "-C", str(dest), "fetch", "--tags"], check=True)
    subprocess.run(["git", "-C", str(dest), "checkout", MCP_BENCH_PINNED_REF], check=True)
    return dest


def parse_server_tasks(server_id: str, source: Path = DEFAULT_DEST) -> list[Any]:
    """Convert MCP-Bench task JSON for one server into `dspy.Example` list.

    Field mapping (per plan v5.1 — Unit 5 documents the exact source fields):
        mb_task["task_id"]          -> Example.task_id
        mb_task["user_query"]       -> Example.user_request    (agent input)
        mb_task["success_criteria"] -> Example.expected_outcome (judge ground truth)
        mb_task                     -> Example.mcp_bench_rubric (preserved for audit)

    Multi-step rubrics collapse to "Task succeeds if ALL of: (a)..., (b)..., (c)..."
    and the `Assess` judge returns pass only if all clauses are satisfied.

    NOTE: stub returns empty list until Day-1 sub-sequence pins the actual
    MCP-Bench task format (which depends on the cloned commit).
    """
    _ = source  # silence unused until live integration
    _ = server_id
    return []
