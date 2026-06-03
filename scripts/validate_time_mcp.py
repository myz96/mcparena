"""Validation passes for the time-mcp Phase 1 result.

Pass 1 (read trajectories): captures full ReAct trajectories on every trial so
we can inspect whether successes are real answers and what failures look like.

Pass 2 (baseline stability): runs the baseline condition N times independently
to measure run-to-run variance — if the baseline drifts from ~0% the +37.5pp
lift is unstable.

Pass 3 (held-out task): runs the GEPA-discovered prompt against a brand-new
hand-written time task (Sydney/Berlin/São Paulo) to test whether the lift
transfers off the train set.

Usage:
    set -a && source .env && set +a
    uv run python scripts/validate_time_mcp.py
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
RESULTS_DIR = REPO_ROOT / "pilot-results" / "validation-time-mcp"
EXISTING_RUN = REPO_ROOT / "pilot-results" / "time-mcp-phase1" / "results.json"
TASKS_PATH = REPO_ROOT / "examples" / "tasks_time_mcp.json"

N_BASELINE_RUNS = 3
N_TRIALS_PER_RUN = 4

HELD_OUT_TASK = {
    "task_id": "time_mcp_heldout_southern",
    "user_request": (
        "A global strategy team needs to decide the best 1-hour call slot "
        "during local business hours (09:00-17:00) in three offices: "
        "Sydney (Australia/Sydney), Berlin (Europe/Berlin), and "
        "São Paulo (America/Sao_Paulo). They have three candidate UTC slots "
        "next week: 03:00 UTC, 13:00 UTC, and 22:00 UTC.\n\n"
        "Steps:\n"
        "1. Fetch the current local time in each office's timezone to confirm "
        "the timezone identifiers resolve correctly.\n"
        "2. For each UTC candidate slot, convert to each office's local time.\n"
        "3. Determine for each office whether the converted local time falls "
        "within business hours (09:00-17:00).\n"
        "4. Count offices in business hours per UTC slot.\n"
        "5. Pick the slot with the highest count (earlier UTC wins ties).\n"
        "6. Output JSON with: utc_slot, sydney_time, berlin_time, sao_paulo_time, "
        "offices_within_business_hours, recommendation (yes/no for the best slot)."
    ),
}


def stamp(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _trial_record(example: Any, pred: Any, score: float) -> dict[str, Any]:
    """Strip down a (example, pred, score) tuple for JSON serialization.

    `dspy.ReAct.forward` stores trajectory as `dict[str, Any]` with keys like
    `thought_0`, `tool_name_0`, `tool_args_0`, `observation_0`, `thought_1`,
    `...` — *not* `list[dict]`. We serialize the dict as-is (after _safe-ing
    its values) so the trajectory is preserved for cross-model re-judging.
    """
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


def _safe(value: Any) -> Any:
    """Best-effort coerce arbitrary objects to JSON-friendly form."""
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_safe(v) for v in value]
    return str(value)


def _run_eval(program: Any, examples: list[Any], n_trials: int) -> Any:
    from mcparena.pilot.judge import judge_metric_evaluate

    trials = [ex for _ in range(n_trials) for ex in examples]
    return dspy.Evaluate(
        devset=trials, metric=judge_metric_evaluate, num_threads=8, failure_score=0.0
    )(program)


def _load_existing_discovered_prompt() -> str:
    data = json.loads(EXISTING_RUN.read_text())
    prompt = data.get("discovered_prompt")
    if not prompt:
        raise RuntimeError(f"No discovered_prompt in {EXISTING_RUN}")
    return str(prompt)


def _build_baseline(tool_list: list[Any]) -> Any:
    return dspy.ReAct("user_request -> final_answer", tools=tool_list, max_iters=20)


def _build_gepa_replica(tool_list: list[Any], discovered_prompt: str) -> Any:
    """Materialize a ReAct program with GEPA's discovered prompt as the signature."""
    program = dspy.ReAct("user_request -> final_answer", tools=tool_list, max_iters=20)
    # Inject the discovered prompt as the instructions on the inner Predict module
    inner = getattr(program, "react", None) or getattr(program, "predict", None)
    if inner is None or not hasattr(inner, "signature"):
        raise RuntimeError("Could not locate ReAct.predict/react.signature to override")
    inner.signature = inner.signature.with_instructions(discovered_prompt)
    return program


def pass_1_and_2_baseline_stability() -> dict[str, Any]:
    """Run baseline N times against time-mcp; capture trajectories each run."""
    from mcp.client.stdio import StdioServerParameters

    from mcparena.optimize import _build_examples
    from mcparena.pilot import costs, tools
    from mcparena.pilot.lm import get_lm

    tasks = json.loads(TASKS_PATH.read_text())
    examples = _build_examples(tasks)
    stdio_params = StdioServerParameters(
        command="uv",
        args=["run", "python", "-m", "mcp_server_time"],
        env={},
        cwd=str(REPO_ROOT / "third_party/mcp-bench-tasks/mcp_servers/time-mcp"),
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
            program = _build_baseline(tool_list)
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
    stamp(
        f"=== baseline stability: mean of means {summary['mean_of_means']:.1%} "
        f"± stddev {summary['stddev_of_means']:.1%} across {N_BASELINE_RUNS} runs ==="
    )
    return summary


def pass_3_held_out_task() -> dict[str, Any]:
    """Run baseline + GEPA-discovered-prompt against a hand-written held-out task."""
    from mcp.client.stdio import StdioServerParameters

    from mcparena.optimize import _build_examples
    from mcparena.pilot import costs, tools
    from mcparena.pilot.lm import get_lm

    examples = _build_examples([HELD_OUT_TASK])
    discovered_prompt = _load_existing_discovered_prompt()
    stdio_params = StdioServerParameters(
        command="uv",
        args=["run", "python", "-m", "mcp_server_time"],
        env={},
        cwd=str(REPO_ROOT / "third_party/mcp-bench-tasks/mcp_servers/time-mcp"),
    )

    program_lm = get_lm()
    dspy.configure(lm=program_lm)

    stamp("=== held-out: baseline ===")
    costs.reset()
    with tools.persistent_session(stdio_params) as session:
        tool_list = tools.make_tools(session)
        baseline_program = _build_baseline(tool_list)
        baseline_eval = _run_eval(baseline_program, examples, N_TRIALS_PER_RUN)
    costs.absorb_lm_history(program_lm, role="program", condition="held_out_baseline")
    baseline_records = [_trial_record(ex, pred, score) for ex, pred, score in baseline_eval.results]
    baseline_mean = sum(r["score"] for r in baseline_records) / max(len(baseline_records), 1)
    baseline_cost = costs.get_state().total_usd
    stamp(f"  held-out baseline mean: {baseline_mean:.1%}, cost ${baseline_cost:.4f}")

    stamp("=== held-out: gepa-discovered prompt ===")
    costs.reset()
    with tools.persistent_session(stdio_params) as session:
        tool_list = tools.make_tools(session)
        gepa_program = _build_gepa_replica(tool_list, discovered_prompt)
        gepa_eval = _run_eval(gepa_program, examples, N_TRIALS_PER_RUN)
    costs.absorb_lm_history(program_lm, role="program", condition="held_out_gepa")
    gepa_records = [_trial_record(ex, pred, score) for ex, pred, score in gepa_eval.results]
    gepa_mean = sum(r["score"] for r in gepa_records) / max(len(gepa_records), 1)
    gepa_cost = costs.get_state().total_usd
    stamp(f"  held-out gepa mean: {gepa_mean:.1%}, cost ${gepa_cost:.4f}")

    delta_pp = (gepa_mean - baseline_mean) * 100
    summary = {
        "held_out_task_id": HELD_OUT_TASK["task_id"],
        "n_trials": N_TRIALS_PER_RUN,
        "baseline_mean": baseline_mean,
        "gepa_mean": gepa_mean,
        "delta_pp": delta_pp,
        "baseline_scores": [r["score"] for r in baseline_records],
        "gepa_scores": [r["score"] for r in gepa_records],
        "cost_usd": round(baseline_cost + gepa_cost, 4),
    }
    (RESULTS_DIR / "held_out").mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "held_out" / "baseline_trajectories.json").write_text(
        json.dumps(baseline_records, indent=2)
    )
    (RESULTS_DIR / "held_out" / "gepa_trajectories.json").write_text(
        json.dumps(gepa_records, indent=2)
    )
    (RESULTS_DIR / "held_out" / "summary.json").write_text(json.dumps(summary, indent=2))
    stamp(f"=== held-out delta: {delta_pp:+.2f}pp ({baseline_mean:.1%} → {gepa_mean:.1%}) ===")
    return summary


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp(f"validation results → {RESULTS_DIR}")
    stability = pass_1_and_2_baseline_stability()
    held_out = pass_3_held_out_task()

    stamp("")
    stamp("=== combined validation summary ===")
    stamp(
        f"baseline (mean of means) {stability['mean_of_means']:.1%} "
        f"± {stability['stddev_of_means']:.1%} over {stability['n_runs']} runs"
    )
    stamp(
        f"held-out: baseline {held_out['baseline_mean']:.1%} → "
        f"gepa {held_out['gepa_mean']:.1%}  (Δ {held_out['delta_pp']:+.2f}pp)"
    )
    stamp(f"total validation cost: ${stability['cumulative_cost_usd'] + held_out['cost_usd']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
