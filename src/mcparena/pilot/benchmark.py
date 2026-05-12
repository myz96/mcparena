"""MCP-Bench (Accenture) task loader.

Pins to a specific MCP-Bench commit. The repo is cloned (gitignored) on
demand; tasks are parsed into `dspy.Example` lists keyed by the
`mcp_bench_id` of each pilot server.

Per plan v5.1 — Day-1 sub-sequence inspected the pinned commit's task format
and locked the field mapping below. MCP-Bench does NOT publish explicit
success-criteria fields; `task_description` is itself the success
specification (a multi-step procedure narrative). We set both
`Example.user_request` and `Example.expected_outcome` to that narrative so
the `Assess` judge has a single source of truth for "what success looks like."
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

MCP_BENCH_REPO = "https://github.com/Accenture/mcp-bench"
MCP_BENCH_PINNED_REF = "7a8eaeae83a842a2949080acc5473f65e1569daf"
DEFAULT_DEST = Path("third_party/mcp-bench-tasks")
SINGLE_TASKS_FILE = "tasks/mcpbench_tasks_single_runner_format.json"


def ensure_mcp_bench_cloned(dest: Path = DEFAULT_DEST) -> Path:
    """Clone or update Accenture/mcp-bench at the pinned ref. Idempotent."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        subprocess.run(["git", "clone", MCP_BENCH_REPO, str(dest)], check=True)
    subprocess.run(["git", "-C", str(dest), "fetch", "--all"], check=True)
    subprocess.run(["git", "-C", str(dest), "checkout", MCP_BENCH_PINNED_REF], check=True)
    return dest


def _load_single_server_tasks(source: Path) -> dict[str, list[dict[str, Any]]]:
    """Load the single-server tasks JSON and index by `server_name`."""
    path = source / SINGLE_TASKS_FILE
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    return {entry["server_name"]: entry["tasks"] for entry in data.get("server_tasks", [])}


def parse_server_tasks(
    mcp_bench_id: str,
    source: Path = DEFAULT_DEST,
) -> list[Any]:
    """Parse MCP-Bench single-server tasks for one server into `dspy.Example` list.

    Field mapping (MCP-Bench -> dspy.Example):
      task_id           -> Example.task_id
      task_description  -> Example.user_request (agent input)
      task_description  -> Example.expected_outcome (same; MCP-Bench has no
                           separate success-criteria field — the description
                           IS the criteria)
      fuzzy_description -> Example.mcp_bench_fuzzy
      dependency_analysis, distraction_servers -> Example.mcp_bench_metadata

    Returns an empty list if the source dir is not yet cloned.
    """
    import dspy  # lazy

    indexed = _load_single_server_tasks(source)
    raw_tasks = indexed.get(mcp_bench_id, [])
    examples: list[Any] = []
    for t in raw_tasks:
        ex = dspy.Example(
            task_id=t["task_id"],
            user_request=t["task_description"],
            expected_outcome=t["task_description"],
            mcp_bench_fuzzy=t.get("fuzzy_description", ""),
            mcp_bench_metadata={
                "dependency_analysis": t.get("dependency_analysis"),
                "distraction_servers": t.get("distraction_servers"),
            },
        ).with_inputs("user_request")
        examples.append(ex)
    return examples
