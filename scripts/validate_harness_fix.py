"""Test the REAL hypothesis after the make_tools schema fix:
with MCP tool arg-schemas now forwarded to the program LM, does convert_time
start succeeding, does baseline jump, and does GEPA still add anything on top?

Runs baseline + GEPA on time-mcp (2 tasks x n_trials), capturing trajectories
so we can directly measure the convert_time error rate (the mechanism) — not
just the task score.

Usage:
    set -a && source .env && set +a
    uv run python scripts/validate_harness_fix.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import dspy

sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "pilot-results" / "validation-harness-fix"
TASKS_PATH = REPO_ROOT / "examples" / "tasks_time_mcp.json"
N_TRIALS = 4


def stamp(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _convert_time_stats(traj: dict[str, Any]) -> tuple[int, int]:
    """Return (convert_time call count, errored count) from a ReAct trajectory dict."""
    calls = errs = 0
    i = 0
    while f"tool_name_{i}" in traj:
        if traj.get(f"tool_name_{i}") == "convert_time":
            calls += 1
            obs = str(traj.get(f"observation_{i}", "")).lower()
            if "error" in obs or "missing required" in obs:
                errs += 1
        i += 1
    return calls, errs


def _summarize(eval_result: Any, label: str) -> dict[str, Any]:
    rows = []
    total_calls = total_errs = 0
    for ex, pred, score in eval_result.results:
        traj = getattr(pred, "trajectory", {}) or {}
        calls, errs = _convert_time_stats(traj if isinstance(traj, dict) else {})
        total_calls += calls
        total_errs += errs
        rows.append(
            {
                "task_id": getattr(ex, "task_id", "?"),
                "score": float(score),
                "convert_time_calls": calls,
                "convert_time_errors": errs,
                "final_answer": str(getattr(pred, "final_answer", ""))[:300],
            }
        )
    mean = sum(r["score"] for r in rows) / max(len(rows), 1)
    err_rate = total_errs / total_calls if total_calls else 0.0
    stamp(
        f"  {label}: score {mean:.1%} ({sum(r['score'] for r in rows):.0f}/{len(rows)}) | "
        f"convert_time {total_errs}/{total_calls} errored ({err_rate:.0%})"
    )
    return {"label": label, "mean_score": mean, "convert_time_error_rate": err_rate, "rows": rows}


def main() -> int:
    from mcp.client.stdio import StdioServerParameters

    from mcparena.optimize import _build_examples
    from mcparena.pilot import costs, tools
    from mcparena.pilot.judge import judge_metric_evaluate, judge_metric_gepa
    from mcparena.pilot.lm import get_lm

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tasks = json.loads(TASKS_PATH.read_text())
    examples = _build_examples(tasks)
    trials = [ex for _ in range(N_TRIALS) for ex in examples]
    stdio = StdioServerParameters(
        command="uv",
        args=["run", "python", "-m", "mcp_server_time"],
        env={},
        cwd=str(REPO_ROOT / "third_party/mcp-bench-tasks/mcp_servers/time-mcp"),
    )

    program_lm = get_lm()
    reflection_lm = get_lm(role="reflection")
    dspy.configure(lm=program_lm)
    costs.reset()

    def _eval(program: Any) -> Any:
        return dspy.Evaluate(
            devset=trials, metric=judge_metric_evaluate, num_threads=8, failure_score=0.0
        )(program)

    stamp("=== POST-FIX: tool arg-schemas now forwarded to the model ===")
    with tools.persistent_session(stdio) as session:
        tool_list = tools.make_tools(session)
        stamp("→ baseline")
        baseline = _summarize(
            _eval(dspy.ReAct("user_request -> final_answer", tools=tool_list, max_iters=20)),
            "baseline",
        )

        stamp("→ gepa (max_full_evals=1)")
        gepa_prog = dspy.ReAct("user_request -> final_answer", tools=tool_list, max_iters=20)
        opt = dspy.GEPA(
            metric=judge_metric_gepa,
            reflection_lm=reflection_lm,
            max_full_evals=1,
            track_stats=True,
        ).compile(gepa_prog, trainset=examples, valset=examples)
        gepa = _summarize(_eval(opt), "gepa")

    inner = getattr(opt, "react", None) or getattr(opt, "predict", None)
    discovered = getattr(getattr(inner, "signature", None), "instructions", None)

    out = {
        "n_trials": N_TRIALS,
        "n_per_condition": len(trials),
        "baseline": baseline,
        "gepa": gepa,
        "discovered_prompt": discovered,
        "cost_usd": round(costs.get_state().total_usd, 4),
    }
    (RESULTS_DIR / "results.json").write_text(json.dumps(out, indent=2))

    stamp("")
    stamp("=== summary (POST-FIX) ===")
    stamp(
        f"baseline: {baseline['mean_score']:.1%}  (convert_time err {baseline['convert_time_error_rate']:.0%})"
    )
    stamp(
        f"gepa:     {gepa['mean_score']:.1%}  (convert_time err {gepa['convert_time_error_rate']:.0%})"
    )
    stamp(f"delta:    {(gepa['mean_score'] - baseline['mean_score']) * 100:+.1f}pp")
    stamp(f"cost:     ${out['cost_usd']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
