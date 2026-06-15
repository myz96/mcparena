"""Audit the Phase 0 verifier: is the 100% baseline real, or is the check lenient?

Runs baseline ReAct on a few held-out tasks with full trajectory capture, and
prints — for manual inspection — the agent's final answer, the ground truth, the
verifier's score+feedback, and the tool-call sequence (to see whether the agent
actually used the converter tools or answered from its own knowledge).
"""

from __future__ import annotations

import json
import sys
from typing import Any

import dspy
from scripts.phase0_unit_converter import _stdio, generate_tasks, verify

sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]


def _tool_seq(pred: Any) -> list[str]:
    traj = getattr(pred, "trajectory", {}) or {}
    seq = []
    i = 0
    while f"tool_name_{i}" in traj:
        seq.append(str(traj.get(f"tool_name_{i}")))
        i += 1
    return seq


def main() -> int:
    from mcparena.pilot import tools
    from mcparena.pilot.lm import get_lm

    dspy.configure(lm=get_lm())
    with tools.persistent_session(_stdio()) as session:
        test_tasks = generate_tasks(session, seed=3, n_single=4, n_multi=12)
        # 2 singles + 2 multis
        sample = test_tasks[:2] + [t for t in test_tasks if t["kind"] == "multi"][:2]
        tool_list = tools.make_tools(session)
        prog = dspy.ReAct("user_request -> final_answer", tools=tool_list, max_iters=25)
        for t in sample:
            pred = prog(user_request=t["user_request"])
            fa = str(getattr(pred, "final_answer", ""))
            score, fb = verify(t, fa)
            print(f"\n===== {t['task_id']} ({t['kind']}) =====")
            print("tool calls:", _tool_seq(pred))
            print("ground_truth:", json.dumps(t["ground_truth"])[:500])
            print("final_answer:", " ".join(fa.split())[:500])
            print(f"verifier: score={score}  feedback={fb}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
