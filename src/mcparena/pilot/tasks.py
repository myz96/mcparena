"""Pilot server specifications and MCP-Bench task collections.

Servers and task IDs are locked from MCP-Bench (Accenture, NeurIPS 2025
Workshop), pinned to a specific commit (see `benchmark.MCP_BENCH_PINNED_REF`).

Three servers chosen by:
1. No-env-key requirement (easy local setup).
2. Domain diversity (computation / knowledge / API audit).
3. Task complexity ranging from deterministic single-step (Math) to
   multi-step trajectory-heavy (OpenAPI Explorer).

Per-server MCP-Bench published baseline scores are NOT broken out in
mcpbench's public leaderboard — overall claude-sonnet-4 score is 0.681.
Per-server difficulty here is estimated from inspecting task descriptions
in the pinned commit (see `notes` field on each spec).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from mcp.client.stdio import StdioServerParameters
from pydantic import BaseModel

Condition = Literal["baseline", "miprov2", "gepa", "axis_ii", "axis_iii"]
DifficultyTier = Literal["easy", "medium", "hard"]
Transport = Literal["stdio", "http"]


def _repo_root() -> Path:
    """Walk up from this file to find the repo root (the dir containing pyproject.toml)."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    # Fallback: return the cwd. The smoke-adapter run validates path resolution.
    return Path.cwd()


# Absolute path to MCP-Bench's server binaries (after `mcp_servers/install.sh`).
# Resolved from __file__ so `mcparena pilot` works regardless of cwd at invocation.
_MCP_BENCH_SERVERS_ROOT = _repo_root() / "third_party/mcp-bench-tasks/mcp_servers"


class ServerSpec(BaseModel):
    """Specification of a pilot MCP server target."""

    name: str
    mcp_bench_id: str
    baseline_score_mcp_bench: float  # 0.0 = not yet measured (per-server scores unpublished)
    difficulty_tier: DifficultyTier
    stratification_rationale: str
    transport: Transport
    command: str
    args: list[str]
    cwd: str | None = None
    env: dict[str, str] = {}
    notes: str = ""

    def to_stdio_params(self) -> StdioServerParameters:
        """Materialize stdio connection parameters from this spec."""
        return StdioServerParameters(
            command=self.command,
            args=self.args,
            env=self.env,
            cwd=self.cwd,
        )


PILOT_SERVERS: list[ServerSpec] = [
    ServerSpec(
        name="math_mcp",
        mcp_bench_id="Math MCP",
        baseline_score_mcp_bench=0.0,  # per-server scores unpublished by MCP-Bench
        difficulty_tier="easy",
        stratification_rationale=(
            "Computational tools (sum, mean, std-dev, percentile, etc.) over "
            "small numeric arrays — deterministic, short trajectories, high "
            "expected baseline. Ceiling test for optimization."
        ),
        transport="stdio",
        command="node",
        args=["build/index.js"],
        cwd=str(_MCP_BENCH_SERVERS_ROOT / "math-mcp"),
        notes="Estimated easy tier from MCP-Bench task inspection (math_mcp_000, math_mcp_001).",
    ),
    ServerSpec(
        name="wikipedia",
        mcp_bench_id="Wikipedia",
        baseline_score_mcp_bench=0.0,
        difficulty_tier="medium",
        stratification_rationale=(
            "Search + extract + synthesize from real-world articles. Multi-step "
            "but bounded; realistic agent usage pattern; moderate baseline."
        ),
        transport="stdio",
        command="uv",
        args=["run", "python", "-m", "wikipedia_mcp"],
        cwd=str(_MCP_BENCH_SERVERS_ROOT / "wikipedia-mcp"),
        notes="Estimated medium tier (wikipedia_000, wikipedia_001).",
    ),
    ServerSpec(
        name="openapi_explorer",
        mcp_bench_id="OpenAPI Explorer",
        baseline_score_mcp_bench=0.0,
        difficulty_tier="hard",
        stratification_rationale=(
            "Multi-step comparative audits across multiple API specifications "
            "(5+ tool calls per task). Long trajectories; biggest headroom for "
            "optimization to show value; estimated hardest of the no-env-key set."
        ),
        transport="stdio",
        command="node",
        args=["index.js", "run"],
        cwd=str(_MCP_BENCH_SERVERS_ROOT / "openapi-mcp-server"),
        notes="Estimated hard tier (openapi_explorer_000, openapi_explorer_001).",
    ),
]


# Per-server task lists populated from MCP-Bench's
# `tasks/mcpbench_tasks_single_runner_format.json` via
# `mcparena.pilot.benchmark.parse_server_tasks`. Empty until the loader runs
# (during pilot main() entry; not at module import time).
TASKS_BY_SERVER: dict[str, list[Any]] = {spec.name: [] for spec in PILOT_SERVERS}
