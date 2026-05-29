"""mcparena CLI entry point."""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

# Long-running pilot prints status per-condition; without line buffering, those
# updates queue up in a 4-8 KB stdout buffer and the user sees nothing until the
# command exits ~20+ minutes later.
sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]
sys.stderr.reconfigure(line_buffering=True)  # type: ignore[union-attr]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mcparena",
        description="Optimization layer over MCP benchmarks via DSPy GEPA + MIPROv2.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    pilot = subparsers.add_parser(
        "pilot",
        help="Run the pilot (validates DSPy improves MCP tool-use)",
    )
    smoke = pilot.add_mutually_exclusive_group()
    smoke.add_argument(
        "--smoke-adapter",
        action="store_true",
        help="Verify GEPA MCP adapter loads on Filesystem (~$1, R9 gate)",
    )
    smoke.add_argument(
        "--smoke-budget",
        action="store_true",
        help="Calibrate cost on 1 server / 1 task / 2 trials (~$8)",
    )
    smoke.add_argument(
        "--shake-out",
        action="store_true",
        help="Single-server full-conditions sanity check (~$30): does our "
        "baseline match MCP-Bench's published score? do optimizers move "
        "the needle at all? Gate before authorizing the full pilot.",
    )
    pilot.add_argument(
        "--server",
        default=None,
        help="Run pilot for one server only (filter; default: all 3)",
    )
    pilot.add_argument(
        "--condition",
        default="all",
        choices=["all", "baseline", "miprov2", "gepa", "axis_ii", "axis_iii"],
        help="Run only a single condition (default: all 5)",
    )
    pilot.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Skip git-clean check for the pre-registration honor system",
    )

    optimize = subparsers.add_parser(
        "optimize",
        help="Optimize ANY MCP server (the distribution wedge)",
        description=(
            "Run baseline + GEPA against an arbitrary MCP server. Outputs the "
            "GEPA-discovered prompt (with any tool-schema quirks GEPA found) "
            "plus a paired bootstrap CI on the lift vs baseline."
        ),
    )
    optimize.add_argument(
        "--server-cmd",
        required=True,
        help='Shell command to launch the MCP server, e.g. "uv run python -m wikipedia_mcp"',
    )
    optimize.add_argument(
        "--server-cwd",
        default=None,
        help="Working directory for the MCP server subprocess (default: inherit)",
    )
    optimize.add_argument(
        "--tasks",
        required=True,
        type=Path,
        help='JSON file containing a list of {"user_request": "..."} task objects',
    )
    optimize.add_argument(
        "--n-trials",
        type=int,
        default=3,
        help="Trials per task per condition (default: 3)",
    )
    optimize.add_argument(
        "--max-full-evals",
        type=int,
        default=1,
        help="GEPA rollout budget (default: 1 for cheap demo; raise for production)",
    )
    optimize.add_argument(
        "--output-dir",
        type=Path,
        default=Path("mcparena-optimize-results"),
        help="Where to write results.json and summary.md",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "pilot":
        from mcparena.pilot import pilot as pilot_module

        return pilot_module.main(args)

    if args.command == "optimize":
        from mcparena.optimize import load_config_from_args, run_optimize

        parts = shlex.split(args.server_cmd)
        if not parts:
            print("error: --server-cmd is empty", file=sys.stderr)
            return 2
        config = load_config_from_args(
            server_cmd=parts[0],
            server_args=parts[1:],
            tasks_path=args.tasks,
            n_trials=args.n_trials,
            output_dir=args.output_dir,
            max_full_evals=args.max_full_evals,
            server_cwd=args.server_cwd,
        )
        run_optimize(config)
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
