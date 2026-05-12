"""MCP-Bench (Accenture) task loader.

Pins to a specific MCP-Bench commit. The repo is cloned (gitignored) on
demand; tasks are parsed into `dspy.Example` lists keyed by the
`mcp_bench_id` of each pilot server.

After `git checkout`, the resolved HEAD SHA is verified to match the pinned
constant. A mismatch (force-push, ref rewrite, etc.) raises and refuses to
run — protects against running attacker-controlled subprocesses from a
tampered clone.
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
    """Clone Accenture/mcp-bench at the pinned ref. Idempotent; verifies SHA.

    Raises ``RuntimeError`` if the resolved HEAD SHA does not equal
    ``MCP_BENCH_PINNED_REF`` after checkout (defends against upstream tamper /
    force-push by refusing to launch MCP servers from an unverified tree).
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        subprocess.run(["git", "clone", MCP_BENCH_REPO, str(dest)], check=True)
    subprocess.run(["git", "-C", str(dest), "fetch", "--all"], check=True)
    subprocess.run(["git", "-C", str(dest), "checkout", MCP_BENCH_PINNED_REF], check=True)

    resolved = subprocess.check_output(
        ["git", "-C", str(dest), "rev-parse", "HEAD"], text=True
    ).strip()
    if resolved != MCP_BENCH_PINNED_REF:
        raise RuntimeError(
            f"MCP-Bench checkout SHA mismatch — expected {MCP_BENCH_PINNED_REF}, "
            f"got {resolved}. Refusing to launch servers from an unverified tree."
        )
    return dest


def _load_single_server_tasks(source: Path) -> dict[str, list[dict[str, Any]]] | None:
    """Load the single-server tasks JSON and index by `server_name`.

    Returns ``None`` if the file is absent (clone not yet run); callers
    distinguish "not cloned" from "cloned but id missing".
    """
    path = source / SINGLE_TASKS_FILE
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return {entry["server_name"]: entry["tasks"] for entry in data.get("server_tasks", [])}


def parse_server_tasks(
    mcp_bench_id: str,
    source: Path = DEFAULT_DEST,
) -> list[Any]:
    """Parse MCP-Bench single-server tasks for one server into `dspy.Example` list.

    Field mapping (MCP-Bench -> dspy.Example):
      task_id           -> Example.task_id
      task_description  -> Example.user_request (agent input AND success criteria)
      fuzzy_description -> Example.mcp_bench_fuzzy (preserved, not used in pilot)
      dependency_analysis, distraction_servers -> Example.mcp_bench_metadata

    Returns an empty list if the source dir is not yet cloned (callers should
    call `ensure_mcp_bench_cloned` first). Raises ``KeyError`` if the source
    IS cloned but the given `mcp_bench_id` is absent — a typo, not a missing
    clone.
    """
    import dspy

    indexed = _load_single_server_tasks(source)
    if indexed is None:
        return []
    if mcp_bench_id not in indexed:
        valid = sorted(indexed.keys())
        raise KeyError(
            f"mcp_bench_id {mcp_bench_id!r} not found in MCP-Bench task file. Valid ids: {valid}"
        )

    raw_tasks = indexed[mcp_bench_id]
    examples: list[Any] = []
    for t in raw_tasks:
        ex = dspy.Example(
            task_id=t["task_id"],
            user_request=t["task_description"],
            mcp_bench_fuzzy=t.get("fuzzy_description", ""),
            mcp_bench_metadata={
                "dependency_analysis": t.get("dependency_analysis"),
                "distraction_servers": t.get("distraction_servers"),
            },
        ).with_inputs("user_request")
        examples.append(ex)
    return examples
