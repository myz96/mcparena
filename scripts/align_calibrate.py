"""Human-alignment calibration — does the verifier agree with a human?

Two-step blind workflow (the human never sees the verifier's verdict or the
ground truth while grading):

  1) prepare: stratified-sample N trials → write a BLIND gradesheet (task +
     agent answer only) for the human to fill, plus a hidden answer key.
  2) score:   read the filled gradesheet, join with the key, compute
     agreement + Cohen's κ + confusion + disagreements, and apply the gate.

Run AFTER a verifiable run has written pilot-results/verifiable/{condition}_trials.jsonl.

Usage:
    uv run python scripts/align_calibrate.py prepare --condition baseline --n 12
    #   …open the gradesheet, write PASS/FAIL after each `GRADE:` …
    uv run python scripts/align_calibrate.py score --condition baseline
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from mcparena.verifiable import align

REPO_ROOT = Path(__file__).resolve().parents[1]
VDIR = REPO_ROOT / "pilot-results" / "verifiable"


def _trials(condition: str) -> list[dict[str, Any]]:
    path = VDIR / f"{condition}_trials.jsonl"
    if not path.exists():
        sys.exit(f"missing {path} — run scripts/verifiable_run.py first")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def prepare(condition: str, n: int, seed: int) -> None:
    sample = align.stratified_sample(_trials(condition), n=n, seed=seed)
    key_path = VDIR / f"_gradekey_{condition}.json"
    sheet_path = VDIR / f"gradesheet_{condition}.md"
    key_path.write_text(json.dumps(sample, indent=2))

    blocks = [
        f"# Alignment gradesheet — condition={condition}  (n={len(sample)})",
        "",
        "For each item: does the agent's answer CORRECTLY complete the task? "
        "Write `PASS` or `FAIL` after `GRADE:`.",
        "Grade blind — the verifier's verdict and ground truth are hidden.",
        "",
    ]
    for i, t in enumerate(sample, 1):
        blocks += [
            f"## item {i}",
            "TASK:",
            t["user_request"],
            "",
            "AGENT ANSWER:",
            t["final_answer"] or "(empty)",
            "",
            "GRADE: ",
            "",
            "---",
            "",
        ]
    sheet_path.write_text("\n".join(blocks))
    print(f"wrote {sheet_path} ({len(sample)} items) + hidden key {key_path.name}")
    print(
        "Fill in each GRADE: PASS/FAIL, then: uv run python scripts/align_calibrate.py score --condition",
        condition,
    )


def score(condition: str) -> None:
    key_path = VDIR / f"_gradekey_{condition}.json"
    sheet_path = VDIR / f"gradesheet_{condition}.md"
    if not key_path.exists() or not sheet_path.exists():
        sys.exit("run `prepare` first")
    sample = json.loads(key_path.read_text())
    grades = [g.upper() for g in re.findall(r"GRADE:\s*(PASS|FAIL)", sheet_path.read_text(), re.I)]
    if len(grades) != len(sample):
        sys.exit(f"found {len(grades)} grades but {len(sample)} items — fill in every GRADE:")

    pairs: list[tuple[int, int]] = []
    disagreements = []
    for t, g in zip(sample, grades, strict=True):
        v = align.verifier_pass(t["score"])
        h = 1 if g == "PASS" else 0
        pairs.append((v, h))
        if v != h:
            disagreements.append(
                {
                    "task_id": t["task_id"],
                    "kind": t["kind"],
                    "verifier": "PASS" if v else "FAIL",
                    "human": g,
                    "verifier_score": t["score"],
                    "ground_truth": t["ground_truth"],
                    "final_answer": (t["final_answer"] or "")[:300],
                }
            )

    report = align.gate(pairs)
    report["condition"] = condition
    report["disagreements"] = disagreements
    (VDIR / f"alignment_{condition}.json").write_text(json.dumps(report, indent=2))

    print(f"\n=== alignment report — condition={condition} (n={report['n']}) ===")
    print(f"agreement:   {report['agreement']:.1%}")
    print(f"cohen kappa: {report['cohen_kappa']}")
    print(f"confusion:   {report['confusion']}")
    print(f"GATE (κ≥0.6 & agree≥0.8): {'PASS ✓' if report['passed_gate'] else 'FAIL ✗'}")
    if report["n"] < 25:
        print(
            f"NOTE: n={report['n']} is a smoke check (wide CI; κ noisy). Bump to ~30 for a real gate."
        )
    if disagreements:
        print(
            f"\n{len(disagreements)} disagreement(s) — inspect: verifier bug / bad task / under-specified rubric?"
        )
        for d in disagreements:
            print(
                f"  - {d['task_id']} ({d['kind']}): verifier={d['verifier']} human={d['human']} | gt={json.dumps(d['ground_truth'])[:120]}"
            )


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--condition", default="baseline")
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    s = sub.add_parser("score")
    s.add_argument("--condition", default="baseline")
    args = ap.parse_args()
    if args.cmd == "prepare":
        prepare(args.condition, args.n, args.seed)
    else:
        score(args.condition)
    return 0


if __name__ == "__main__":
    sys.exit(main())
