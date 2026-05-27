"""Phase 2: tighten the CI on the +16.7pp Phase 1 GEPA signal (wikipedia).

Re-runs baseline AND GEPA both at n_trials=8 (= 16 paired evals) so the
bootstrap CI is computed on matched, larger samples. Writes a separate
phase2.json so the original shake-out checkpoint stays intact.

Cost expectation at Qwen3 rates: ~$15-30 total (baseline ~$2, GEPA ~$15-25
with max_full_evals=1).

Usage:
    set -a && source .env && set +a
    uv run python scripts/run_phase2.py
"""

from __future__ import annotations

import json
import sys
import time

sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]


def stamp(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


N_TRIALS = 8


def main() -> int:
    from mcparena.pilot import costs
    from mcparena.pilot.pilot import (
        RESULTS_DIR,
        aggregate_and_report,
        run_baseline,
        run_gepa,
    )

    stamp(f"=== Phase 2: baseline + GEPA on wikipedia at n_trials={N_TRIALS} ===")
    costs.reset()

    stamp("→ baseline")
    t0 = time.time()
    try:
        baseline = run_baseline("wikipedia", n_trials=N_TRIALS)
    except Exception as exc:
        stamp(f"✗ baseline raised: {type(exc).__name__}: {exc}")
        return 1
    stamp(f"baseline {time.time() - t0:.0f}s — mean {baseline.get('mean_score', 0):.1f}%")

    # Persist after baseline so a mid-GEPA crash doesn't lose the rerun.
    aggregate_and_report({"wikipedia": {"baseline": baseline}}, mode="phase2")

    stamp("→ gepa (max_full_evals=1, n_trials=8)")
    t0 = time.time()
    try:
        gepa = run_gepa("wikipedia", n_trials=N_TRIALS, max_full_evals=1)
    except Exception as exc:
        stamp(f"✗ gepa raised: {type(exc).__name__}: {exc}")
        # Still persist what we have so baseline isn't lost.
        aggregate_and_report({"wikipedia": {"baseline": baseline}}, mode="phase2")
        return 1
    stamp(f"gepa {time.time() - t0:.0f}s — mean {gepa.get('mean_score', 0):.1f}%")

    aggregate_and_report({"wikipedia": {"baseline": baseline, "gepa": gepa}}, mode="phase2")

    # Re-read the just-written file to surface the bootstrap-CI delta.
    summary = json.loads((RESULTS_DIR / "phase2.json").read_text())
    delta = summary["servers"]["wikipedia"]["deltas"].get("gepa", {})

    state = costs.get_state()
    stamp("")
    stamp("=== Phase 2 summary ===")
    stamp(f"baseline mean: {baseline.get('mean_score', 0):.2f}% (n={baseline.get('n_trials', 0)})")
    stamp(f"gepa     mean: {gepa.get('mean_score', 0):.2f}% (n={gepa.get('n_trials', 0)})")
    stamp(f"delta:         {delta.get('delta_pp', 0):+.2f}pp")
    stamp(
        f"95% CI:        [{delta.get('ci_low_pp', 0):+.2f}pp, "
        f"{delta.get('ci_high_pp', 0):+.2f}pp] (n_paired={delta.get('n_paired', 0)})"
    )
    stamp(f"phase 2 cost:  ${state.total_usd:.4f} (Qwen3 rates)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
