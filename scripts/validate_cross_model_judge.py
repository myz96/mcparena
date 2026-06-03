"""Validation Pass 4: re-judge saved trajectories with a cross-model judge.

The pilot used same-model judging (Qwen3 grades Qwen3). This script re-judges
each saved (user_request, trajectory, final_answer) record with a different
provider — Claude Sonnet 4 — and computes the agreement rate between the
two judges. If agreement is high (>80%), our same-model judge is not biased
in a way that materially distorts the result; if low (<70%), the lift may
be a judging artifact.

Reads trajectories saved by `scripts/validate_time_mcp.py` under
`pilot-results/validation-time-mcp/`. Writes a JSON summary to
`pilot-results/validation-time-mcp/cross_model_judge.json`.

Usage:
    set -a && source .env && set +a
    uv run python scripts/validate_cross_model_judge.py
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
TASKS_PATH = REPO_ROOT / "examples" / "tasks_time_mcp.json"
CROSS_JUDGE_MODEL = os.environ.get(
    "MCPARENA_CROSS_JUDGE_MODEL", "openrouter/anthropic/claude-sonnet-4"
)


def stamp(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _build_cross_judge() -> Any:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set; cross-model judge needs it")
    cross_lm = dspy.LM(
        model=CROSS_JUDGE_MODEL,
        api_key=api_key,
        api_base="https://openrouter.ai/api/v1",
        temperature=0.0,  # deterministic re-judging
        max_tokens=2048,
        cache=False,
    )
    return cross_lm


def _user_request_for(task_id: str) -> str:
    tasks = json.loads(TASKS_PATH.read_text())
    for t in tasks:
        if t.get("task_id") == task_id:
            return str(t["user_request"])
    # Fall back to held-out task if id matches
    if task_id == "time_mcp_heldout_southern":
        from scripts.validate_time_mcp import HELD_OUT_TASK

        return str(HELD_OUT_TASK["user_request"])
    raise KeyError(f"unknown task_id: {task_id}")


def _trajectory_to_str(trajectory: dict[str, Any] | list[dict[str, Any]]) -> str:
    """Render a saved trajectory into the text form dspy.ReAct produces.

    Handles both shapes the validation script may have saved:
    - dict (current dspy.ReAct format): keys like `thought_0`, `tool_name_0`, ...
    - list[dict] (legacy / hypothetical): list of step dicts.
    """
    if isinstance(trajectory, dict):
        return "\n".join(f"{k}: {v}" for k, v in trajectory.items())
    if isinstance(trajectory, list):
        lines: list[str] = []
        for i, step in enumerate(trajectory):
            if isinstance(step, dict):
                for k, v in step.items():
                    lines.append(f"step {i} | {k}: {v}")
        return "\n".join(lines)
    return ""


def _judge_one(
    cross_lm: Any,
    user_request: str,
    trajectory: dict[str, Any] | list[dict[str, Any]],
    final_answer: str,
) -> tuple[float, str]:
    judge = dspy.Predict(Assess)
    with dspy.context(lm=cross_lm):
        result = judge(
            user_request=user_request,
            trajectory=_trajectory_to_str(trajectory),
            final_answer=final_answer or "",
        )
    return (1.0 if result.success else 0.0, result.reason)


def _gather_records() -> list[tuple[str, str, dict[str, Any]]]:
    """Return (source_label, condition, record) tuples for every saved trial."""
    out: list[tuple[str, str, dict[str, Any]]] = []
    for run_dir in sorted(VAL_DIR.glob("baseline_run_*")):
        records = json.loads((run_dir / "trajectories.json").read_text())
        for rec in records:
            out.append((run_dir.name, "baseline", rec))
    held_out = VAL_DIR / "held_out"
    if (held_out / "baseline_trajectories.json").exists():
        for rec in json.loads((held_out / "baseline_trajectories.json").read_text()):
            out.append(("held_out", "baseline", rec))
    if (held_out / "gepa_trajectories.json").exists():
        for rec in json.loads((held_out / "gepa_trajectories.json").read_text()):
            out.append(("held_out", "gepa", rec))
    return out


def main() -> int:
    if not VAL_DIR.exists():
        stamp(f"validation dir not found: {VAL_DIR}")
        stamp("run `scripts/validate_time_mcp.py` first")
        return 1

    cross_lm = _build_cross_judge()
    stamp(f"cross-model judge: {CROSS_JUDGE_MODEL}")

    records = _gather_records()
    stamp(f"re-judging {len(records)} saved trial records")

    judgements: list[dict[str, Any]] = []
    for i, (source, condition, rec) in enumerate(records):
        try:
            user_request = _user_request_for(rec["task_id"])
        except KeyError as exc:
            stamp(f"  skip [{i + 1}/{len(records)}]: {exc}")
            continue
        try:
            cross_score, reason = _judge_one(
                cross_lm,
                user_request=user_request,
                trajectory=rec.get("trajectory", []),
                final_answer=str(rec.get("final_answer", "")),
            )
        except Exception as exc:
            stamp(f"  ✗ [{i + 1}/{len(records)}] {source}/{condition}: {type(exc).__name__}: {exc}")
            continue
        original_score = float(rec.get("score", 0.0))
        agrees = bool(int(cross_score) == int(original_score))
        judgements.append(
            {
                "source": source,
                "condition": condition,
                "task_id": rec.get("task_id"),
                "original_score": original_score,
                "cross_score": cross_score,
                "agrees": agrees,
                "cross_reason": reason,
            }
        )
        marker = "✓" if agrees else "✗"
        stamp(
            f"  {marker} [{i + 1}/{len(records)}] {source}/{condition} {rec.get('task_id')}: "
            f"orig {original_score} vs cross {cross_score}"
        )

    if not judgements:
        stamp("no judgements collected")
        return 1

    agreements = sum(1 for j in judgements if j["agrees"])
    by_condition: dict[str, dict[str, int]] = {}
    for j in judgements:
        bucket = by_condition.setdefault(j["condition"], {"n": 0, "agree": 0})
        bucket["n"] += 1
        if j["agrees"]:
            bucket["agree"] += 1

    summary = {
        "cross_judge_model": CROSS_JUDGE_MODEL,
        "n_records": len(judgements),
        "n_agreements": agreements,
        "overall_agreement_rate": agreements / len(judgements),
        "by_condition": {
            cond: {
                **bucket,
                "agreement_rate": bucket["agree"] / bucket["n"] if bucket["n"] else 0.0,
            }
            for cond, bucket in by_condition.items()
        },
        "judgements": judgements,
    }
    out_path = VAL_DIR / "cross_model_judge.json"
    out_path.write_text(json.dumps(summary, indent=2))
    stamp("")
    stamp("=== cross-model judge summary ===")
    stamp(
        f"overall agreement: {summary['overall_agreement_rate']:.1%} "
        f"({agreements}/{len(judgements)})"
    )
    for cond, bucket in summary["by_condition"].items():
        stamp(f"  {cond}: {bucket['agreement_rate']:.1%} ({bucket['agree']}/{bucket['n']})")
    stamp(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
