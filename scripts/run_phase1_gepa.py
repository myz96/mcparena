"""Phase 1: run GEPA only on wikipedia, append to shake-out checkpoint.

Skips miprov2 (already errored on prior run from Sonnet-rate cost cap),
runs GEPA with smoke-tier budget (max_full_evals=1) for a cheap first
signal before scaling up.

Cost expectation with corrected Qwen3 pricing: a few cents to a few dollars.

Usage:
    set -a && source .env && set +a
    uv run python scripts/run_phase1_gepa.py
"""

from __future__ import annotations

import json
import sys
import time

sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]


def stamp(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> int:
    from mcparena.pilot import costs
    from mcparena.pilot.pilot import RESULTS_DIR, aggregate_and_report, run_gepa

    stamp("=== Phase 1: GEPA on wikipedia, max_full_evals=1, n_trials=3 ===")

    costs.reset()
    t0 = time.time()
    try:
        result = run_gepa("wikipedia", n_trials=3, max_full_evals=1)
    except Exception as exc:
        stamp(f"✗ GEPA raised: {type(exc).__name__}: {exc}")
        return 1
    stamp(f"GEPA finished in {time.time() - t0:.1f}s")

    path = RESULTS_DIR / "shake-out.json"
    existing = json.loads(path.read_text())
    existing["raw_results"]["wikipedia"]["gepa"] = result
    aggregate_and_report({"wikipedia": existing["raw_results"]["wikipedia"]}, mode="shake-out")

    state = costs.get_state()
    stamp(f"phase 1 cost: ${state.total_usd:.4f} (Qwen3 rates)")
    stamp(f"  by condition: {dict(state.by_condition)}")
    stamp(f"GEPA mean: {result.get('mean_score', 0):.2f}% " f"({result.get('n_trials', 0)} trials)")
    stamp(f"baseline mean: {existing['raw_results']['wikipedia']['baseline']['mean_score']:.2f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
