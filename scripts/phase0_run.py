"""Phase 0 run: verifiable baseline vs GEPA on auto-generated unit-converter tasks.

The whole point: a PROGRAMMATIC verifier is the metric — NO judge LM, so no
judge noise, no format-gating. GEPA gets rich, true feedback from the verifier.
Train/val/test split; baseline and GEPA are compared on the HELD-OUT test set.

Decision gate: does GEPA lift the verifiable baseline on held-out multi-step tasks?

Usage:
    set -a && source .env && set +a
    uv run python scripts/phase0_run.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import dspy
from scripts.phase0_unit_converter import _stdio, generate_tasks, verify

sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "pilot-results" / "phase0-unit-converter"
MAX_FULL_EVALS = 6


def stamp(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _examples(tasks: list[dict[str, Any]]) -> list[Any]:
    return [
        dspy.Example(user_request=t["user_request"], task_obj=t).with_inputs("user_request")
        for t in tasks
    ]


def _tool_errs(pred: Any) -> tuple[int, int]:
    traj = getattr(pred, "trajectory", {}) or {}
    if not isinstance(traj, dict):
        return 0, 0
    c = e = 0
    i = 0
    while f"tool_name_{i}" in traj:
        c += 1
        if "error" in str(traj.get(f"observation_{i}", "")).lower():
            e += 1
        i += 1
    return c, e


def _summarize(eval_result: Any, tasks: list[dict[str, Any]], label: str) -> dict[str, Any]:
    by_kind: dict[str, list[float]] = {}
    tc = te = 0
    for (_ex, pred, score), t in zip(eval_result.results, tasks, strict=False):
        by_kind.setdefault(t["kind"], []).append(float(score))
        c, e = _tool_errs(pred)
        tc += c
        te += e
    overall = [s for v in by_kind.values() for s in v]
    mean = sum(overall) / len(overall) if overall else 0.0
    kind_means = {k: round(sum(v) / len(v), 3) for k, v in by_kind.items()}
    stamp(
        f"  {label}: overall {mean:.1%} (n={len(overall)}) by_kind={kind_means} "
        f"| tool-call errors {te}/{tc}"
    )
    return {"label": label, "mean": mean, "by_kind": kind_means, "n": len(overall)}


def main() -> int:
    from mcparena.pilot import costs, tools
    from mcparena.pilot.lm import get_lm

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    program_lm = get_lm()
    reflection_lm = get_lm(role="reflection")
    dspy.configure(lm=program_lm)
    costs.reset()

    def metric_eval(example: Any, pred: Any, trace: Any = None) -> float:
        return verify(example.task_obj, getattr(pred, "final_answer", ""))[0]

    def metric_gepa(
        gold: Any, pred: Any, trace: Any = None, pred_name: Any = None, pred_trace: Any = None
    ) -> Any:
        score, fb = verify(gold.task_obj, getattr(pred, "final_answer", ""))
        return dspy.Prediction(score=score, feedback=fb)

    stamp("=== Phase 0: verifiable baseline vs GEPA (unit-converter, server-as-oracle) ===")
    with tools.persistent_session(_stdio()) as session:
        stamp("generating verifiable tasks (train/val/test) by executing against the server…")
        train_tasks = generate_tasks(session, seed=1, n_single=2, n_multi=12)
        val_tasks = generate_tasks(session, seed=2, n_single=1, n_multi=7)
        test_tasks = generate_tasks(session, seed=3, n_single=4, n_multi=12)
        stamp(
            f"  train={len(train_tasks)} val={len(val_tasks)} test={len(test_tasks)} "
            f"(multi-weighted; singles are a saturation check)"
        )
        trainset, valset, testset = (
            _examples(train_tasks),
            _examples(val_tasks),
            _examples(test_tasks),
        )

        tool_list = tools.make_tools(session)

        def _eval(prog: Any) -> Any:
            return dspy.Evaluate(
                devset=testset, metric=metric_eval, num_threads=8, failure_score=0.0
            )(prog)

        stamp("→ baseline (vanilla ReAct, verifiable metric)")
        baseline_prog = dspy.ReAct("user_request -> final_answer", tools=tool_list, max_iters=25)
        baseline = _summarize(_eval(baseline_prog), test_tasks, "baseline")
        costs.absorb_lm_history(program_lm, role="program", condition="baseline")

        stamp(f"→ gepa (max_full_evals={MAX_FULL_EVALS}, verifier feedback, train/val split)")
        gepa_prog = dspy.ReAct("user_request -> final_answer", tools=tool_list, max_iters=25)
        opt = dspy.GEPA(
            metric=metric_gepa,
            reflection_lm=reflection_lm,
            max_full_evals=MAX_FULL_EVALS,
            num_threads=8,
            track_stats=True,
        ).compile(gepa_prog, trainset=trainset, valset=valset)
        gepa = _summarize(_eval(opt), test_tasks, "gepa")
        costs.absorb_lm_history(reflection_lm, role="reflection", condition="gepa")
        costs.absorb_lm_history(program_lm, role="program", condition="gepa")

    inner = getattr(opt, "react", None) or getattr(opt, "predict", None)
    discovered = getattr(getattr(inner, "signature", None), "instructions", None)
    state = costs.get_state()
    out = {
        "max_full_evals": MAX_FULL_EVALS,
        "n_train": len(train_tasks),
        "n_val": len(val_tasks),
        "n_test": len(test_tasks),
        "baseline": baseline,
        "gepa": gepa,
        "delta_pp": round((gepa["mean"] - baseline["mean"]) * 100, 2),
        "discovered_prompt": discovered,
        "cost_usd": round(state.total_usd, 4),
    }
    (RESULTS_DIR / "results.json").write_text(json.dumps(out, indent=2))

    stamp("")
    stamp("=== Phase 0 summary (held-out test set, VERIFIABLE metric — no judge LM) ===")
    stamp(f"baseline: {baseline['mean']:.1%}  by_kind={baseline['by_kind']}")
    stamp(f"gepa:     {gepa['mean']:.1%}  by_kind={gepa['by_kind']}")
    stamp(f"delta:    {out['delta_pp']:+.2f}pp")
    stamp(f"cost:     ${out['cost_usd']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
