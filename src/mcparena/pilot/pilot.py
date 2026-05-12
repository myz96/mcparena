"""mcparena pilot runner — orchestrates the 5-condition spike.

This module is the entrypoint for `mcparena pilot ...`. Structural in Unit 3;
live behavior wires up in subsequent units. Each `run_*` function is a stub that
documents the intended `dspy.Evaluate` / `dspy.MIPROv2` / `dspy.GEPA` calls.

Cost discipline (see plan §Cost Model):
- baseline / axis_ii / axis_iii  : Sonnet 4.6 trials only        (~$11 total)
- miprov2                        : MIPROv2(auto="light") compile (~$25 total)
- gepa                           : GEPA(auto="light") + Opus reflection (~$82 total)
- smoke-adapter                  : ~$1 — R9 gate
- smoke-budget                   : ~$10 — calibrates per-server cost projections
- hard cap                       : $350 (runner halts at $300 cumulative)
- reflection share gate          : drop GEPA on remaining servers if Opus > 60%
                                   of total spend after first GEPA run
"""

from __future__ import annotations

import argparse
from typing import Any

from mcparena.pilot.tasks import PILOT_SERVERS, Condition


def make_tools(session: Any) -> list[Any]:
    """Wrap MCP server tools as `dspy.Tool` callables.

    Uses the closure pattern (no MCPAdapter class) — each tool becomes a thin
    sync wrapper around `asyncio.run(session.call_tool(name, kwargs))`.
    Implementation deferred until live behavior wires up.
    """
    raise NotImplementedError("Unit 3 stub — implementation in subsequent unit")


def permute_tools(tools: list[Any], max_permutations: int = 4) -> list[list[Any]]:
    """For axis (ii): generate up to N permutations of the tool list."""
    raise NotImplementedError("Unit 3 stub")


def inject_one_shot(tools: list[Any], exemplar: dict[str, Any]) -> list[Any]:
    """For axis (iii): inject a worked example into each tool's description."""
    raise NotImplementedError("Unit 3 stub")


def run_baseline(server_id: str, n_trials: int = 5) -> Any:
    """Vanilla `dspy.ReAct` + `dspy.Evaluate(failure_score=0.0, num_threads=8)`."""
    raise NotImplementedError("Unit 3 stub")


def run_miprov2(server_id: str, n_trials: int = 5) -> Any:
    """`dspy.MIPROv2(auto="light")`.compile -> re-evaluate."""
    raise NotImplementedError("Unit 3 stub")


def run_gepa(server_id: str, n_trials: int = 5) -> Any:
    """`dspy.GEPA(auto="light", reflection_lm=opus-4-7, max_tokens=16000)`.compile."""
    raise NotImplementedError("Unit 3 stub")


def run_axis_ii(server_id: str, n_trials: int = 5) -> list[Any]:
    """Brute-force ≤4 permutations of the tool list; return all + best."""
    raise NotImplementedError("Unit 3 stub")


def run_axis_iii(server_id: str, n_trials: int = 5) -> Any:
    """Post-process baseline best trajectory into tool descriptions; re-evaluate."""
    raise NotImplementedError("Unit 3 stub")


def run_smoke_adapter() -> int:
    """R9 gate: verify GEPA MCP adapter loads + accepts our config (~$1)."""
    raise NotImplementedError("Unit 3 stub")


def run_smoke_budget() -> int:
    """Calibrate cost on 1 server / 1 task / 2 trials / all 5 conditions (~$8)."""
    raise NotImplementedError("Unit 3 stub")


def run_shake_out(server_id: str = "math_mcp") -> int:
    """Single-server full-conditions sanity check (~$30).

    Question this tier answers: does our baseline number actually match
    MCP-Bench's published score for this server, and do any of the 4
    optimization conditions produce non-zero deltas? Gate before authorizing
    the full pilot ($300 cap).

    Scope: 1 server (default: easy tier = Math MCP) × both MCP-Bench tasks ×
    3 trials × all 5 conditions = 30 trial-equivalents + 1 MIPROv2 compile +
    1 GEPA compile. Estimated cost ~$30.
    """
    raise NotImplementedError("Unit 3 stub")


def check_cost_caps(cumulative_usd: float, reflection_share: float) -> None:
    """Halt runner at $300 cumulative OR reflection share >60%."""
    raise NotImplementedError("Unit 3 stub")


def aggregate_and_report(results: dict[str, dict[Condition, Any]]) -> None:
    """Inline `scipy.stats.bootstrap` per (server, condition) vs baseline.

    Writes:
    - pilot-results/summary.json (per-server / per-condition deltas + CIs)
    - pilot-results/memo.md      (1-page memo, populated from one of three
                                  pre-written narrative frames based on results)
    """
    raise NotImplementedError("Unit 3 stub")


def main(args: argparse.Namespace) -> int:
    """CLI entrypoint dispatched from `mcparena.cli.main`."""
    if args.smoke_adapter:
        return run_smoke_adapter()
    if args.smoke_budget:
        return run_smoke_budget()
    if args.shake_out:
        # Default to the easy-tier server; --server flag can override.
        return run_shake_out(server_id=args.server or "math_mcp")

    # Full pilot: iterate (server, condition) per pre-registration sample size.
    # Verifies pre-reg first (Unit 5), respects --server / --condition filters.
    raise NotImplementedError(
        "Full pilot run wires up in subsequent unit; "
        f"requested args={args}, servers={[s.name for s in PILOT_SERVERS]}"
    )
