"""mcparena CLI entry point."""

from __future__ import annotations

import argparse
import sys


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

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "pilot":
        from mcparena.pilot import pilot as pilot_module

        return pilot_module.main(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
