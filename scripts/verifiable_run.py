"""Verifiable baseline (and optional GEPA) over hard unit-converter tasks.

The verifier IS the metric — no judge LM. Persists per-trial outputs
(task, agent answer, verdict) to pilot-results/verifiable/ so the
human-alignment tool (align_calibrate.py) can grade them.

Usage:
    set -a && source .env && set +a
    uv run python scripts/verifiable_run.py baseline      # A2 difficulty calibration
    uv run python scripts/verifiable_run.py gepa          # A3 the GEPA gate
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import dspy

from mcparena.verifiable.tasks import generate_tasks, uc_stdio_params, verify


def _failing_sensors(task: dict[str, Any], final_answer: str) -> set[str]:
    """Names of sensors the verifier marked wrong (parsed from feedback)."""
    _, fb = verify(task, final_answer)
    return set(re.findall(r"sensor_\d+", fb))


sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "pilot-results" / "verifiable"
MAX_FULL_EVALS = 6
# Weak AGENT (headroom to optimize) + strong REFLECTION (GEPA proposes good prompts).
# The strong model saturates these tasks at ~98%, so optimization is only testable
# on an agent that genuinely struggles — which is also where prompt-opt matters.
PROGRAM_MODEL = os.environ.get("MCPARENA_PROGRAM_MODEL", "openrouter/qwen/qwen3-8b")
REFLECTION_MODEL = os.environ.get(
    "MCPARENA_REFLECTION_MODEL", "openrouter/qwen/qwen3-235b-a22b-2507"
)
# Agent temperature. 0.7 default; set 0 to test whether per-instance metric noise
# (which breaks GEPA's accept/reject) is driven by agent stochasticity.
AGENT_TEMP = float(os.environ.get("MCPARENA_AGENT_TEMP", "0.7"))
STABILITY_RUNS = 2
# All multi_hard: the only kind with real headroom (high-volume careful execution).
# chain/aggregate/conditional saturate even for an 8B agent once tools are wired.
TEST_COUNTS = {"multi_hard": 8}
TRAIN_COUNTS = {"multi_hard": 8}
VAL_COUNTS = {"multi_hard": 4}


def stamp(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _examples(tasks: list[dict[str, Any]]) -> list[Any]:
    return [
        dspy.Example(user_request=t["user_request"], task_obj=t).with_inputs("user_request")
        for t in tasks
    ]


def _persist_trials(
    eval_result: Any, tasks: list[dict[str, Any]], condition: str, path: Path
) -> None:
    with path.open("a") as fh:
        for (_ex, pred, score), t in zip(eval_result.results, tasks, strict=False):
            fa = str(getattr(pred, "final_answer", ""))
            _, fb = verify(t, fa)
            fh.write(
                json.dumps(
                    {
                        "condition": condition,
                        "task_id": t["task_id"],
                        "kind": t["kind"],
                        "score": float(score),
                        "feedback": fb,
                        "final_answer": fa,
                        "ground_truth": t["ground_truth"],
                        "user_request": t["user_request"],
                    }
                )
                + "\n"
            )


def _summarize(eval_result: Any, tasks: list[dict[str, Any]], label: str) -> dict[str, Any]:
    by_kind: dict[str, list[float]] = {}
    for (_ex, _pred, score), t in zip(eval_result.results, tasks, strict=False):
        by_kind.setdefault(t["kind"], []).append(float(score))
    overall = [s for v in by_kind.values() for s in v]
    mean = sum(overall) / len(overall) if overall else 0.0
    kind_means = {k: round(sum(v) / len(v), 2) for k, v in by_kind.items()}
    stamp(f"  {label}: overall {mean:.1%} (n={len(overall)}) by_kind={kind_means}")
    return {"label": label, "mean": mean, "by_kind": kind_means, "n": len(overall)}


def main() -> int:
    from mcparena.pilot import costs, tools
    from mcparena.pilot.lm import get_lm

    mode = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    program_lm = get_lm(model_id=PROGRAM_MODEL, temperature=AGENT_TEMP)
    reflection_lm = get_lm(model_id=REFLECTION_MODEL, role="reflection")
    dspy.configure(lm=program_lm)
    costs.reset()
    stamp(f"agent(program)={PROGRAM_MODEL} temp={AGENT_TEMP}  reflection={REFLECTION_MODEL}")

    def metric_eval(example: Any, pred: Any, trace: Any = None) -> float:
        return verify(example.task_obj, getattr(pred, "final_answer", ""))[0]

    def metric_gepa(gold: Any, pred: Any, trace: Any = None, pn: Any = None, pt: Any = None) -> Any:
        score, fb = verify(gold.task_obj, getattr(pred, "final_answer", ""))
        return dspy.Prediction(score=score, feedback=fb)

    stamp(f"=== verifiable run: mode={mode} (hard unit-converter tasks, verifier metric) ===")
    with tools.persistent_session(uc_stdio_params(REPO_ROOT)) as session:
        stamp("generating hard verifiable tasks (executing against the server)…")
        test_tasks = generate_tasks(session, seed=103, counts=TEST_COUNTS)
        testset = _examples(test_tasks)
        tool_list = tools.make_tools(session)

        def _eval(prog: Any) -> Any:
            return dspy.Evaluate(
                devset=testset, metric=metric_eval, num_threads=8, failure_score=0.0
            )(prog)

        if mode == "stability":
            runs: list[dict[str, tuple[float, set[str]]]] = []
            for r in range(STABILITY_RUNS):
                stamp(f"→ stability run {r + 1}/{STABILITY_RUNS} (temp={AGENT_TEMP})")
                prog = dspy.ReAct("user_request -> final_answer", tools=tool_list, max_iters=30)
                ev = _eval(prog)
                rec: dict[str, tuple[float, set[str]]] = {}
                for (_e, pred, score), t in zip(ev.results, test_tasks, strict=False):
                    fa = str(getattr(pred, "final_answer", ""))
                    rec[t["task_id"]] = (float(score), _failing_sensors(t, fa))
                runs.append(rec)
                stamp(f"  run {r + 1}: mean {sum(s for s, _ in rec.values()) / len(rec):.1%}")
            costs.absorb_lm_history(program_lm, role="program", condition="stability")

            stamp("")
            stamp("=== stability: per-task scores across runs + failing-sensor overlap ===")
            jaccards = []
            for t in test_tasks:
                tid = t["task_id"]
                scores = [runs[i][tid][0] for i in range(STABILITY_RUNS)]
                fails = [runs[i][tid][1] for i in range(STABILITY_RUNS)]
                union = set().union(*fails)
                inter = set(fails[0]).intersection(*fails[1:])
                jac = len(inter) / len(union) if union else 1.0
                jaccards.append(jac)
                stamp(
                    f"  {tid}: scores={[round(s, 2) for s in scores]} fails_overlap={jac:.2f} "
                    f"union_fail={sorted(union)}"
                )
            overall = [
                sum(runs[i][t["task_id"]][0] for t in test_tasks) / len(test_tasks)
                for i in range(STABILITY_RUNS)
            ]
            stamp("")
            stamp(
                f"per-run overall: {[f'{m:.1%}' for m in overall]}  avg={sum(overall) / len(overall):.1%}"
            )
            stamp(
                f"mean failing-sensor Jaccard: {sum(jaccards) / len(jaccards):.2f} "
                "(1.0 = same failures each run = deterministic/systematic → GEPA can climb; "
                "~0 = random → noise blocks GEPA)"
            )
            stamp(f"cost: ${costs.get_state().total_usd:.4f}")
            return 0

        stamp("→ baseline (vanilla ReAct, verifier metric)")
        baseline_prog = dspy.ReAct("user_request -> final_answer", tools=tool_list, max_iters=30)
        baseline_eval = _eval(baseline_prog)
        baseline = _summarize(baseline_eval, test_tasks, "baseline")
        _persist_trials(baseline_eval, test_tasks, "baseline", OUT_DIR / "baseline_trials.jsonl")
        costs.absorb_lm_history(program_lm, role="program", condition="baseline")

        gepa = None
        discovered = None
        if mode == "gepa":
            stamp("generating train/val…")
            train_tasks = generate_tasks(session, seed=101, counts=TRAIN_COUNTS)
            val_tasks = generate_tasks(session, seed=102, counts=VAL_COUNTS)
            stamp(f"→ gepa (max_full_evals={MAX_FULL_EVALS}, verifier feedback)")
            gepa_prog = dspy.ReAct("user_request -> final_answer", tools=tool_list, max_iters=30)
            opt = dspy.GEPA(
                metric=metric_gepa,
                reflection_lm=reflection_lm,
                max_full_evals=MAX_FULL_EVALS,
                num_threads=8,
                track_stats=True,
            ).compile(gepa_prog, trainset=_examples(train_tasks), valset=_examples(val_tasks))
            gepa_eval = _eval(opt)
            gepa = _summarize(gepa_eval, test_tasks, "gepa")
            _persist_trials(gepa_eval, test_tasks, "gepa", OUT_DIR / "gepa_trials.jsonl")
            costs.absorb_lm_history(reflection_lm, role="reflection", condition="gepa")
            costs.absorb_lm_history(program_lm, role="program", condition="gepa")
            inner = getattr(opt, "react", None) or getattr(opt, "predict", None)
            discovered = getattr(getattr(inner, "signature", None), "instructions", None)

    state = costs.get_state()
    out = {
        "mode": mode,
        "n_test": len(test_tasks),
        "baseline": baseline,
        "gepa": gepa,
        "delta_pp": round((gepa["mean"] - baseline["mean"]) * 100, 2) if gepa else None,
        "discovered_prompt": discovered,
        "cost_usd": round(state.total_usd, 4),
    }
    (OUT_DIR / f"{mode}_summary.json").write_text(json.dumps(out, indent=2))

    stamp("")
    stamp(f"=== summary (mode={mode}, VERIFIABLE metric — no judge LM) ===")
    stamp(f"baseline: {baseline['mean']:.1%}  by_kind={baseline['by_kind']}")
    if gepa:
        stamp(f"gepa:     {gepa['mean']:.1%}  by_kind={gepa['by_kind']}")
        stamp(f"delta:    {out['delta_pp']:+.2f}pp")
    stamp(f"cost:     ${out['cost_usd']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
