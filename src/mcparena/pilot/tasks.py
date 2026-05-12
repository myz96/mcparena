"""Pilot server specifications and MCP-Bench task collections.

`PILOT_SERVERS` and `TASKS_BY_SERVER` are PLACEHOLDERS until Unit 5's Day-1
sub-sequence picks 3 stratified servers from MCP-Bench (Accenture) based on
published baseline scores for Claude Sonnet 4.6. The current entries document
the intended shape only.

Five pilot conditions exposed via the `Condition` literal:
- baseline   : vanilla dspy.ReAct, no optimization
- miprov2    : dspy.MIPROv2(auto="light") on program signature
- gepa       : dspy.GEPA(auto="light", reflection_lm=opus-4-7) on program
- axis_ii    : brute-force tool-list permutation search wrapping baseline
- axis_iii   : hand-injected 1-shot example in each tool description
"""

from __future__ import annotations

from typing import Any, Literal

from mcp.client.stdio import StdioServerParameters
from pydantic import BaseModel

Condition = Literal["baseline", "miprov2", "gepa", "axis_ii", "axis_iii"]
DifficultyTier = Literal["easy", "medium", "hard"]
Transport = Literal["stdio", "http"]


class ServerSpec(BaseModel):
    """Specification of a pilot MCP server target."""

    name: str
    mcp_bench_id: str
    baseline_score_mcp_bench: float  # pinned from MCP-Bench leaderboard for Sonnet 4.6
    difficulty_tier: DifficultyTier
    stratification_rationale: str
    transport: Transport
    command: str
    args: list[str]
    env: dict[str, str] = {}
    notes: str = ""

    def to_stdio_params(self) -> StdioServerParameters:
        """Materialize stdio connection parameters from this spec."""
        return StdioServerParameters(
            command=self.command,
            args=self.args,
            env=self.env,
        )


# PLACEHOLDER — populated during Unit 5 Day-1 sub-sequence.
# Three stratified servers: one easy / one medium / one hard for Sonnet 4.6
# per MCP-Bench's published leaderboard.
PILOT_SERVERS: list[ServerSpec] = [
    ServerSpec(
        name="filesystem",
        mcp_bench_id="<TBD>",
        baseline_score_mcp_bench=0.0,
        difficulty_tier="easy",
        stratification_rationale="PLACEHOLDER — confirm after MCP-Bench leaderboard review",
        transport="stdio",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", "pilot-sandbox/filesystem"],
        notes="Anthropic reference server; ceiling test",
    ),
    ServerSpec(
        name="<server_medium>",
        mcp_bench_id="<TBD>",
        baseline_score_mcp_bench=0.0,
        difficulty_tier="medium",
        stratification_rationale="PLACEHOLDER",
        transport="stdio",
        command="<TBD>",
        args=[],
        notes="PLACEHOLDER",
    ),
    ServerSpec(
        name="<server_hard>",
        mcp_bench_id="<TBD>",
        baseline_score_mcp_bench=0.0,
        difficulty_tier="hard",
        stratification_rationale="PLACEHOLDER",
        transport="stdio",
        command="<TBD>",
        args=[],
        notes="PLACEHOLDER",
    ),
]


# Per-server task lists. Populated from `mcparena.pilot.benchmark.parse_server_tasks`
# during Unit 5; stored here as a flat dict keyed by ServerSpec.name.
# Empty until benchmark loader integration.
TASKS_BY_SERVER: dict[str, list[Any]] = {spec.name: [] for spec in PILOT_SERVERS}
