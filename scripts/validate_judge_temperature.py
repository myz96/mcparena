"""Confirm the root-cause: re-judge saved trajectories to decompose the metric noise.

The pilot judge ran at temperature 0.7 (non-deterministic) with the `Assess`
rubric, which also gates on output format. This script re-judges every saved
trajectory THREE ways, all with the SAME model (Qwen3-235b) as the pilot:

  A. temp=0, Assess rubric           (pass 1)
  B. temp=0, Assess rubric           (pass 2)  -> A vs B = determinism check
  C. temp=0, content-only rubric                -> vs A = format-gating effect

Comparisons:
  - original (temp 0.7, Assess) vs A         : temperature noise
  - A vs B                                    : is temp=0 actually deterministic?
  - A vs C                                    : how many fails are pure format-gating?
  - C pass rate                               : the TRUE content-correct rate

Reads the saved trajectories under pilot-results/validation-time-mcp/.

Usage:
    set -a && source .env && set +a
    uv run python scripts/validate_judge_temperature.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import dspy

from mcparena.pilot.judge import Assess

sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]

REPO_ROOT = Path(__file__).resolve().parents[1]
VAL_DIR = REPO_ROOT / "pilot-results" / "validation-time-mcp"
TIME_TASKS = REPO_ROOT / "examples" / "tasks_time_mcp.json"
JUDGE_MODEL = os.environ.get("MCPARENA_JUDGE_MODEL", "openrouter/qwen/qwen3-235b-a22b-2507")


class AssessContentOnly(dspy.Signature):  # type: ignore[misc]
    """Judge ONLY whether the final answer is factually correct for the task.

    IGNORE all formatting, output schema, JSON shape, key names, wrapper
    objects, field naming (e.g. "no" vs false vs "No"), ordering, and
    presentation style. Two answers with the same substantive content MUST
    receive the same verdict regardless of how they are structured. Return
    success=True if the substantive answer is factually correct and complete
    with respect to the task's actual question — even if the JSON shape,
    key names, or wording differ from any format implied by the request.
    Only return success=False if the substantive content is wrong, missing,
    or the agent failed to determine the answer.
    """

    user_request: str = dspy.InputField(
        desc="The user's task. Treat its QUESTION as the success criteria, not its format hints."
    )
    trajectory: str = dspy.InputField(
        desc="Full ReAct trajectory (reasoning + tool calls + outputs)"
    )
    final_answer: str = dspy.InputField()
    success: bool = dspy.OutputField()
    reason: str = dspy.OutputField()


def stamp(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _build_judge_lm(temperature: float) -> Any:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set")
    return dspy.LM(
        model=JUDGE_MODEL,
        api_key=api_key,
        api_base="https://openrouter.ai/api/v1",
        temperature=temperature,
        max_tokens=2048,
        cache=False,
    )


def _user_request_for(task_id: str) -> str:
    for t in json.loads(TIME_TASKS.read_text()):
        if t.get("task_id") == task_id:
            return str(t["user_request"])
    if task_id == "time_mcp_heldout_southern":
        from scripts.validate_time_mcp import HELD_OUT_TASK

        return str(HELD_OUT_TASK["user_request"])
    raise KeyError(task_id)


def _traj_to_str(trajectory: Any) -> str:
    if isinstance(trajectory, dict):
        return "\n".join(f"{k}: {v}" for k, v in trajectory.items())
    if isinstance(trajectory, list):
        return "\n".join(
            f"step {i} | {k}: {v}"
            for i, step in enumerate(trajectory)
            if isinstance(step, dict)
            for k, v in step.items()
        )
    return ""


def _judge(lm: Any, signature: Any, rec: dict[str, Any]) -> float:
    pred = dspy.Predict(signature)
    with dspy.context(lm=lm):
        result = pred(
            user_request=_user_request_for(rec["task_id"]),
            trajectory=_traj_to_str(rec.get("trajectory", [])),
            final_answer=str(rec.get("final_answer", "")),
        )
    return 1.0 if result.success else 0.0


def _gather() -> list[tuple[str, dict[str, Any]]]:
    out: list[tuple[str, dict[str, Any]]] = []
    for run_dir in sorted(VAL_DIR.glob("baseline_run_*")):
        for rec in json.loads((run_dir / "trajectories.json").read_text()):
            out.append((f"{run_dir.name}/baseline", rec))
    held = VAL_DIR / "held_out"
    for cond in ("baseline", "gepa"):
        p = held / f"{cond}_trajectories.json"
        if p.exists():
            for rec in json.loads(p.read_text()):
                out.append((f"held_out/{cond}", rec))
    return out


def main() -> int:
    if not VAL_DIR.exists():
        stamp(f"missing {VAL_DIR}; run validate_time_mcp.py first")
        return 1

    lm_t0 = _build_judge_lm(0.0)
    records = _gather()
    stamp(f"re-judging {len(records)} trajectories with {JUDGE_MODEL}")
    stamp("  A=temp0/Assess  B=temp0/Assess(repeat)  C=temp0/content-only")

    rows: list[dict[str, Any]] = []
    for i, (src, rec) in enumerate(records):
        try:
            a = _judge(lm_t0, Assess, rec)
            b = _judge(lm_t0, Assess, rec)
            c = _judge(lm_t0, AssessContentOnly, rec)
        except Exception as exc:
            stamp(f"  ✗ [{i + 1}/{len(records)}] {src}: {type(exc).__name__}: {exc}")
            continue
        orig = float(rec.get("score", 0.0))
        rows.append({"src": src, "task_id": rec["task_id"], "orig": orig, "a": a, "b": b, "c": c})
        stamp(
            f"  [{i + 1}/{len(records)}] {src} {rec['task_id']}: "
            f"orig={int(orig)} A={int(a)} B={int(b)} C={int(c)}"
        )

    if not rows:
        stamp("no rows")
        return 1

    n = len(rows)
    orig_pass = sum(r["orig"] for r in rows)
    a_pass = sum(r["a"] for r in rows)
    c_pass = sum(r["c"] for r in rows)
    det_agree = sum(1 for r in rows if r["a"] == r["b"])
    temp_flips_up = sum(1 for r in rows if r["orig"] == 0 and r["a"] == 1)
    temp_flips_dn = sum(1 for r in rows if r["orig"] == 1 and r["a"] == 0)
    fmt_flips_up = sum(1 for r in rows if r["a"] == 0 and r["c"] == 1)

    summary = {
        "judge_model": JUDGE_MODEL,
        "n": n,
        "orig_temp07_assess_pass_rate": orig_pass / n,
        "temp0_assess_pass_rate": a_pass / n,
        "temp0_content_only_pass_rate": c_pass / n,
        "determinism_AB_agreement": det_agree / n,
        "temperature_flips_fail_to_pass": temp_flips_up,
        "temperature_flips_pass_to_fail": temp_flips_dn,
        "format_flips_fail_to_pass": fmt_flips_up,
        "rows": rows,
    }
    (VAL_DIR / "judge_temperature.json").write_text(json.dumps(summary, indent=2))

    stamp("")
    stamp("=== judge decomposition ===")
    stamp(f"original (temp 0.7, Assess):      {orig_pass / n:5.1%}  ({int(orig_pass)}/{n})")
    stamp(f"temp 0, Assess:                   {a_pass / n:5.1%}  ({int(a_pass)}/{n})")
    stamp(f"temp 0, content-only (TRUE rate): {c_pass / n:5.1%}  ({int(c_pass)}/{n})")
    stamp("")
    stamp(f"determinism (A==B):               {det_agree / n:5.1%}  ({det_agree}/{n})")
    stamp("  (temp 0.7 judge was non-deterministic; temp 0 should be ~100%)")
    stamp(f"temperature flips fail->pass:     {temp_flips_up}")
    stamp(f"temperature flips pass->fail:     {temp_flips_dn}")
    stamp(f"format-gating fails (A=0 -> C=1): {fmt_flips_up}")
    stamp(f"wrote {VAL_DIR / 'judge_temperature.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
