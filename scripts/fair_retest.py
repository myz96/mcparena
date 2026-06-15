"""Fair re-test of the key hypothesis: does GEPA improve MCP tool use?

Removes every confound the forensic analysis found:
  1. harness bug (tool schemas hidden)  -> FIXED in make_tools (committed)
  2. judge non-determinism (temp 0.7)   -> judge LM at temperature 0
  3. judge gates on format not content  -> content-focused rubric (AssessContent)
  4. GEPA starved of signal/budget      -> GEPA optimizes against the clean judge,
                                            max_full_evals bumped 1 -> 3
  5. tiny n                             -> n_trials=5

Pre-registered design (written before running):
  - Servers: the pinned MCP-Bench tasks for `time` and `wikipedia` (math is
    saturated post-fix; openapi infeasible at 256K). No hand-authored tasks.
  - Conditions: baseline (vanilla dspy.ReAct) vs GEPA.
  - Metric: content-correct success per (server, task), judged by Qwen3 at temp 0
    with an explicitly format-agnostic rubric. SAME metric drives GEPA's search.
  - Report: per-task baseline vs GEPA, tool-call error rate, GEPA discovered
    prompt. Whatever it shows, it stands — this is the honest test.

Usage:
    set -a && source .env && set +a
    uv run python scripts/fair_retest.py time
    uv run python scripts/fair_retest.py wikipedia
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import dspy

sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "pilot-results" / "fair-retest"
SERVERS_ROOT = REPO_ROOT / "third_party/mcp-bench-tasks/mcp_servers"
JUDGE_MODEL = os.environ.get("MCPARENA_JUDGE_MODEL", "openrouter/qwen/qwen3-235b-a22b-2507")

N_TRIALS = 5
GEPA_MAX_FULL_EVALS = 3

SERVERS: dict[str, dict[str, Any]] = {
    "time": {
        "cmd": "uv",
        "args": ["run", "python", "-m", "mcp_server_time"],
        "cwd": str(SERVERS_ROOT / "time-mcp"),
        "tasks": REPO_ROOT / "examples/tasks_time_mcp.json",
    },
    "wikipedia": {
        "cmd": "uv",
        "args": ["run", "python", "-m", "wikipedia_mcp"],
        "cwd": str(SERVERS_ROOT / "wikipedia-mcp"),
        "tasks": REPO_ROOT / "examples/tasks_wikipedia_mcpbench.json",
    },
    # Local computation, NO external API (so no rate limits — unlike wikipedia),
    # 16 tools + multi-step batch+cross-validate tasks => genuine tool-selection
    # difficulty and real headroom. The best clean test of GEPA's value.
    "unit_converter": {
        "cmd": "uv",
        "args": ["run", "unit-converter-mcp"],
        "cwd": str(SERVERS_ROOT / "unit-converter-mcp"),
        "tasks": REPO_ROOT / "examples/tasks_unit_converter.json",
    },
}


class AssessContent(dspy.Signature):  # type: ignore[misc]
    """Judge whether the agent's final answer is factually CORRECT for the task.

    Grade substance ONLY. IGNORE output formatting, JSON shape, key names,
    wrapper objects, field naming (e.g. "no" vs false), ordering, and wording.
    Two answers with the same factual content MUST get the same verdict
    regardless of structure. Return success=True iff the substantive answer is
    correct and complete with respect to the task's actual question. Return
    success=False only if the content is wrong, incomplete, or the agent failed
    to determine the answer.
    """

    user_request: str = dspy.InputField(
        desc="The task. Treat its QUESTION as the success criteria, not its format hints."
    )
    trajectory: str = dspy.InputField(
        desc="Full ReAct trajectory (reasoning + tool calls + outputs)"
    )
    final_answer: str = dspy.InputField()
    success: bool = dspy.OutputField()
    reason: str = dspy.OutputField()


def stamp(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _build_judge_lm() -> Any:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set")
    return dspy.LM(
        model=JUDGE_MODEL,
        api_key=api_key,
        api_base="https://openrouter.ai/api/v1",
        temperature=0.0,
        max_tokens=2048,
        cache=False,
    )


def _make_metrics(judge_lm: Any) -> tuple[Any, Any]:
    """Return (3-arg metric for Evaluate, 5-arg metric for GEPA), both clean-judged."""

    def _core(example: Any, pred: Any) -> tuple[float, str]:
        with dspy.context(lm=judge_lm):
            r = dspy.Predict(AssessContent)(
                user_request=example.user_request,
                trajectory=str(getattr(pred, "trajectory", "")),
                final_answer=getattr(pred, "final_answer", ""),
            )
        return (1.0 if r.success else 0.0, r.reason)

    def metric_eval(example: Any, pred: Any, trace: Any = None) -> float:
        return _core(example, pred)[0]

    def metric_gepa(
        gold: Any, pred: Any, trace: Any = None, pred_name: Any = None, pred_trace: Any = None
    ) -> Any:
        s, reason = _core(gold, pred)
        return dspy.Prediction(score=s, feedback=reason)

    return metric_eval, metric_gepa


def _tool_error_rate(traj: Any) -> tuple[int, int]:
    if not isinstance(traj, dict):
        return 0, 0
    calls = errs = 0
    i = 0
    while f"tool_name_{i}" in traj:
        calls += 1
        obs = str(traj.get(f"observation_{i}", "")).lower()
        if "error" in obs or "missing required" in obs:
            errs += 1
        i += 1
    return calls, errs


def _summarize(eval_result: Any, label: str) -> dict[str, Any]:
    rows = []
    tc = te = 0
    for ex, pred, score in eval_result.results:
        traj = getattr(pred, "trajectory", {}) or {}
        c, e = _tool_error_rate(traj)
        tc += c
        te += e
        rows.append(
            {
                "task_id": getattr(ex, "task_id", "?"),
                "score": float(score),
                "final_answer": str(getattr(pred, "final_answer", ""))[:240],
            }
        )
    mean = sum(r["score"] for r in rows) / max(len(rows), 1)
    by_task: dict[str, list[float]] = {}
    for r in rows:
        by_task.setdefault(r["task_id"], []).append(r["score"])
    pertask = {k: f"{int(sum(v))}/{len(v)}" for k, v in sorted(by_task.items())}
    stamp(
        f"  {label}: {mean:.1%} {pertask} | tool-call errors {te}/{tc} "
        f"({(te / tc if tc else 0):.0%})"
    )
    return {"label": label, "mean": mean, "per_task": pertask, "tool_err": (te, tc), "rows": rows}


def main() -> int:
    from mcp.client.stdio import StdioServerParameters

    from mcparena.optimize import _build_examples
    from mcparena.pilot import costs, tools
    from mcparena.pilot.lm import get_lm

    server = sys.argv[1] if len(sys.argv) > 1 else "time"
    if server not in SERVERS:
        stamp(f"unknown server {server!r}; choose from {list(SERVERS)}")
        return 1
    cfg = SERVERS[server]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    tasks = json.loads(Path(cfg["tasks"]).read_text())
    examples = _build_examples(tasks)
    trials = [ex for _ in range(N_TRIALS) for ex in examples]
    stdio = StdioServerParameters(command=cfg["cmd"], args=cfg["args"], env={}, cwd=cfg["cwd"])

    program_lm = get_lm()
    reflection_lm = get_lm(role="reflection")
    judge_lm = _build_judge_lm()
    metric_eval, metric_gepa = _make_metrics(judge_lm)
    dspy.configure(lm=program_lm)
    costs.reset()

    def _eval(prog: Any) -> Any:
        return dspy.Evaluate(devset=trials, metric=metric_eval, num_threads=8, failure_score=0.0)(
            prog
        )

    stamp(
        f"=== FAIR RE-TEST: {server} (fixed harness + temp-0 content judge + GEPA mfe={GEPA_MAX_FULL_EVALS}) ==="
    )
    stamp(f"  {len(examples)} tasks x n_trials={N_TRIALS} = {len(trials)} eval trials/condition")
    with tools.persistent_session(stdio) as session:
        tool_list = tools.make_tools(session)
        stamp("→ baseline")
        baseline = _summarize(
            _eval(dspy.ReAct("user_request -> final_answer", tools=tool_list, max_iters=20)),
            "baseline",
        )

        stamp(f"→ gepa (max_full_evals={GEPA_MAX_FULL_EVALS}, clean judge as signal)")
        gepa_prog = dspy.ReAct("user_request -> final_answer", tools=tool_list, max_iters=20)
        opt = dspy.GEPA(
            metric=metric_gepa,
            reflection_lm=reflection_lm,
            max_full_evals=GEPA_MAX_FULL_EVALS,
            track_stats=True,
        ).compile(gepa_prog, trainset=examples, valset=examples)
        gepa = _summarize(_eval(opt), "gepa")

    inner = getattr(opt, "react", None) or getattr(opt, "predict", None)
    discovered = getattr(getattr(inner, "signature", None), "instructions", None)
    out = {
        "server": server,
        "n_trials": N_TRIALS,
        "gepa_max_full_evals": GEPA_MAX_FULL_EVALS,
        "judge": f"{JUDGE_MODEL} @ temp 0, content-only rubric",
        "baseline": baseline,
        "gepa": gepa,
        "delta_pp": (gepa["mean"] - baseline["mean"]) * 100,
        "discovered_prompt": discovered,
        "cost_usd": round(costs.get_state().total_usd, 4),
    }
    (RESULTS_DIR / f"{server}.json").write_text(json.dumps(out, indent=2))

    stamp("")
    stamp(f"=== {server} fair-retest summary ===")
    stamp(f"baseline: {baseline['mean']:.1%} {baseline['per_task']}")
    stamp(f"gepa:     {gepa['mean']:.1%} {gepa['per_task']}")
    stamp(f"delta:    {out['delta_pp']:+.1f}pp")
    stamp(f"cost:     ${out['cost_usd']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
