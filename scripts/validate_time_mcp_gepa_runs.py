"""Run `mcparena optimize` against time-mcp N independent times and pool the
GEPA-vs-baseline lift across runs.

Pass 2 showed baseline is noisy (0–37.5% across runs). GEPA was only measured
once per server. To estimate the AVERAGE GEPA lift we need multiple GEPA
observations too, then pool. Each run produces a fresh discovered prompt;
we record them all so we can also check whether GEPA reliably finds a useful
rewrite or sometimes a bad one.

Usage:
    set -a && source .env && set +a
    uv run python scripts/validate_time_mcp_gepa_runs.py
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "pilot-results" / "validation-time-mcp-gepa"
TASKS_PATH = REPO_ROOT / "examples" / "tasks_time_mcp.json"

N_RUNS = 3
N_TRIALS = 4
MAX_FULL_EVALS = 1


def stamp(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> int:
    from mcparena.optimize import OptimizeConfig, run_optimize

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tasks = json.loads(TASKS_PATH.read_text())

    per_run: list[dict[str, object]] = []
    for run_idx in range(N_RUNS):
        stamp(f"=== GEPA run {run_idx + 1}/{N_RUNS} ===")
        out_dir = RESULTS_DIR / f"gepa_run_{run_idx + 1}"
        config = OptimizeConfig(
            server_cmd="uv",
            server_args=["run", "python", "-m", "mcp_server_time"],
            tasks=tasks,
            n_trials=N_TRIALS,
            output_dir=out_dir,
            max_full_evals=MAX_FULL_EVALS,
            server_cwd=str(REPO_ROOT / "third_party/mcp-bench-tasks/mcp_servers/time-mcp"),
        )
        t0 = time.time()
        summary = run_optimize(config)
        elapsed = time.time() - t0
        baseline_scores = summary["baseline"]["per_trial_scores"]
        gepa_scores = summary["gepa"]["per_trial_scores"]
        per_run.append(
            {
                "run": run_idx + 1,
                "baseline_mean": summary["baseline"]["mean_score"],
                "gepa_mean": summary["gepa"]["mean_score"],
                "baseline_scores": baseline_scores,
                "gepa_scores": gepa_scores,
                "delta": summary["delta"],
                "cost_usd": summary["cost"]["total_usd"],
                "elapsed_sec": round(elapsed, 1),
                "discovered_prompt_preview": str(summary["discovered_prompt"] or "")[:300],
            }
        )
        stamp(
            f"  run {run_idx + 1}: baseline {summary['baseline']['mean_score']:.1%} → "
            f"gepa {summary['gepa']['mean_score']:.1%} "
            f"({elapsed:.0f}s, ${summary['cost']['total_usd']:.4f})"
        )

    # Pool across runs
    pooled_baseline: list[float] = [
        s
        for r in per_run
        for s in r["baseline_scores"]  # type: ignore[attr-defined]
    ]
    pooled_gepa: list[float] = [s for r in per_run for s in r["gepa_scores"]]  # type: ignore[attr-defined]
    pooled_baseline_mean = sum(pooled_baseline) / max(len(pooled_baseline), 1)
    pooled_gepa_mean = sum(pooled_gepa) / max(len(pooled_gepa), 1)
    pooled_delta_pp = (pooled_gepa_mean - pooled_baseline_mean) * 100

    baseline_means = [float(r["baseline_mean"]) for r in per_run]
    gepa_means = [float(r["gepa_mean"]) for r in per_run]
    baseline_stddev = statistics.stdev(baseline_means) if len(baseline_means) > 1 else 0.0
    gepa_stddev = statistics.stdev(gepa_means) if len(gepa_means) > 1 else 0.0

    total_cost = sum(float(r["cost_usd"]) for r in per_run)  # type: ignore[arg-type]
    summary = {
        "n_runs": N_RUNS,
        "n_trials_per_run": N_TRIALS * len(tasks),
        "pooled_baseline_n": len(pooled_baseline),
        "pooled_gepa_n": len(pooled_gepa),
        "pooled_baseline_mean": pooled_baseline_mean,
        "pooled_gepa_mean": pooled_gepa_mean,
        "pooled_delta_pp": pooled_delta_pp,
        "per_run_baseline_means": baseline_means,
        "per_run_gepa_means": gepa_means,
        "per_run_baseline_stddev": baseline_stddev,
        "per_run_gepa_stddev": gepa_stddev,
        "per_run": per_run,
        "total_cost_usd": round(total_cost, 4),
    }
    out_path = RESULTS_DIR / "pooled_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    stamp("")
    stamp("=== pooled across runs ===")
    stamp(
        f"baseline: {pooled_baseline_mean:.1%} "
        f"(pooled n={len(pooled_baseline)}, per-run stddev {baseline_stddev:.1%})"
    )
    stamp(
        f"gepa:     {pooled_gepa_mean:.1%} "
        f"(pooled n={len(pooled_gepa)}, per-run stddev {gepa_stddev:.1%})"
    )
    stamp(f"pooled delta: {pooled_delta_pp:+.2f}pp")
    stamp(f"total cost: ${total_cost:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
