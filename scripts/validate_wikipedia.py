"""Wikipedia baseline-stability validation.

Time-MCP validation showed baselines swing 0–37.5% across runs, so the
Wikipedia Phase 2 "0/16 baseline" claim needs the same multi-run check.
Runs baseline 3× independently with n_trials=4 (2 tasks × 4 = 8 trials
per run); captures full ReAct trajectories for later cross-model judging.

Does NOT re-run GEPA — that's an additional cost we'd only spend if
baseline stability is dramatically different from the Phase 2 reading.

Usage:
    set -a && source .env && set +a
    uv run python scripts/validate_wikipedia.py
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import dspy

sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "pilot-results" / "validation-wikipedia"
WIKI_TASKS = REPO_ROOT / "examples" / "tasks_wikipedia.json"

N_BASELINE_RUNS = 3
N_TRIALS_PER_RUN = 4


def stamp(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_safe(v) for v in value]
    return str(value)


def _trial_record(example: Any, pred: Any, score: float) -> dict[str, Any]:
    trajectory = getattr(pred, "trajectory", None)
    if isinstance(trajectory, dict):
        clean_traj: dict[str, Any] | list[dict[str, Any]] = {
            k: _safe(v) for k, v in trajectory.items()
        }
    elif isinstance(trajectory, list):
        clean_traj = [
            {k: _safe(v) for k, v in step.items()}
            if isinstance(step, dict)
            else {"_step": str(step)}
            for step in trajectory
        ]
    else:
        clean_traj = []
    return {
        "task_id": getattr(example, "task_id", "?"),
        "score": score,
        "final_answer": _safe(getattr(pred, "final_answer", "")),
        "trajectory": clean_traj,
    }


def _run_eval(program: Any, examples: list[Any], n_trials: int) -> Any:
    from mcparena.pilot.judge import judge_metric_evaluate

    trials = [ex for _ in range(n_trials) for ex in examples]
    return dspy.Evaluate(
        devset=trials, metric=judge_metric_evaluate, num_threads=8, failure_score=0.0
    )(program)


def main() -> int:
    from mcp.client.stdio import StdioServerParameters

    from mcparena.optimize import _build_examples
    from mcparena.pilot import costs, tools
    from mcparena.pilot.lm import get_lm

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp(f"validation results → {RESULTS_DIR}")

    if not WIKI_TASKS.exists():
        wiki_tasks_inline = [
            {
                "task_id": "wikipedia_000",
                "user_request": (
                    "You are analyzing renewable energy adoption patterns. "
                    "For each of Solar energy and Wind power on Wikipedia: "
                    "(1) check whether a section discussing 'Environmental impact' or "
                    "'Environmental impacts' exists in the article; "
                    "(2) using get_related_topics, identify policy-related topics "
                    "(e.g. 'Feed-in tariff', 'Renewable energy policy', or similar); "
                    "(3) using get_links, verify which (if any) policy article appears in "
                    "the outbound links of BOTH the solar and wind energy articles. "
                    "(4) From the related topics, recommend exactly one additional "
                    "technology to highlight, with rationale. Output JSON with keys "
                    "solar_links_policy_present, wind_links_policy_present, "
                    "recommended_tech, recommendation_rationale."
                ),
            },
        ]
        WIKI_TASKS.write_text(json.dumps(wiki_tasks_inline, indent=2))
        stamp(f"wrote default wikipedia tasks to {WIKI_TASKS}")

    tasks = json.loads(WIKI_TASKS.read_text())
    examples = _build_examples(tasks)
    stdio_params = StdioServerParameters(
        command="uv",
        args=["run", "python", "-m", "wikipedia_mcp"],
        env={},
        cwd=str(REPO_ROOT / "third_party/mcp-bench-tasks/mcp_servers/wikipedia-mcp"),
    )

    program_lm = get_lm()
    dspy.configure(lm=program_lm)

    per_run: list[dict[str, Any]] = []
    cumulative_cost = 0.0

    for run_idx in range(N_BASELINE_RUNS):
        stamp(f"=== baseline run {run_idx + 1}/{N_BASELINE_RUNS} ===")
        costs.reset()
        with tools.persistent_session(stdio_params) as session:
            tool_list = tools.make_tools(session)
            program = dspy.ReAct("user_request -> final_answer", tools=tool_list, max_iters=20)
            eval_result = _run_eval(program, examples, N_TRIALS_PER_RUN)
        costs.absorb_lm_history(program_lm, role="program", condition="baseline")

        records = [_trial_record(ex, pred, score) for ex, pred, score in eval_result.results]
        mean = sum(r["score"] for r in records) / max(len(records), 1)
        run_cost = costs.get_state().total_usd
        cumulative_cost += run_cost
        stamp(f"  baseline {run_idx + 1}: mean {mean:.1%}, cost ${run_cost:.4f}")

        run_dir = RESULTS_DIR / f"baseline_run_{run_idx + 1}"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "trajectories.json").write_text(json.dumps(records, indent=2))
        per_run.append(
            {
                "run": run_idx + 1,
                "mean_score": mean,
                "n_trials": len(records),
                "scores": [r["score"] for r in records],
                "cost_usd": round(run_cost, 4),
            }
        )

    means = [r["mean_score"] for r in per_run]
    stddev = statistics.stdev(means) if len(means) > 1 else 0.0
    summary = {
        "n_runs": N_BASELINE_RUNS,
        "n_trials_per_run": N_TRIALS_PER_RUN * len(examples),
        "mean_of_means": sum(means) / len(means),
        "stddev_of_means": stddev,
        "per_run": per_run,
        "cumulative_cost_usd": round(cumulative_cost, 4),
    }
    (RESULTS_DIR / "baseline_stability.json").write_text(json.dumps(summary, indent=2))
    stamp("")
    stamp(
        f"=== wiki baseline stability: mean of means {summary['mean_of_means']:.1%} "
        f"± stddev {summary['stddev_of_means']:.1%} across {N_BASELINE_RUNS} runs ==="
    )
    stamp(f"total cost: ${summary['cumulative_cost_usd']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
