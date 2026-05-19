"""Single baseline trial with full transcript dump.

Runs ONE math_mcp baseline task end-to-end. Captures and prints:
  - The user request
  - Every tool call (name + args + output) — from pred.trajectory
  - The agent's final answer
  - The judge's verdict + reason

Use this to diagnose all-zero scores: is it judge bias, agent format error,
or genuine task difficulty?

Usage:
    set -a && source .env && set +a
    uv run python scripts/debug_one_baseline.py [server_id] [task_index]
"""

from __future__ import annotations

import sys
import time

sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]


def stamp(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> int:
    server_id = sys.argv[1] if len(sys.argv) > 1 else "math_mcp"
    task_idx = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    stamp(f"=== debug ONE baseline trial: {server_id} task[{task_idx}] ===")

    import dspy

    from mcparena.pilot import tools
    from mcparena.pilot.benchmark import parse_server_tasks
    from mcparena.pilot.judge import Assess
    from mcparena.pilot.lm import get_lm
    from mcparena.pilot.tasks import PILOT_SERVERS

    program_lm = get_lm()
    dspy.configure(lm=program_lm)
    stamp(f"  model: {program_lm.model}")

    spec = next(s for s in PILOT_SERVERS if s.name == server_id)
    examples = parse_server_tasks(spec.mcp_bench_id)
    if not examples:
        stamp(f"  ✗ no tasks loaded for {server_id}")
        return 1
    ex = examples[task_idx]
    stamp(f"  task: {ex.task_id}")
    stamp("")
    stamp("=== USER REQUEST (full) ===")
    print(ex.user_request)
    print()

    with tools.persistent_session(spec.to_stdio_params()) as session:
        stamp(f"  opened session, {len(session.tool_specs)} tools available")
        tool_list = tools.make_tools(session)
        program = dspy.ReAct("user_request -> final_answer", tools=tool_list, max_iters=20)

        stamp("=== AGENT RUNNING (max_iters=20) ===")
        t0 = time.time()
        try:
            pred = program(user_request=ex.user_request)
        except Exception as exc:
            stamp(f"  ✗ agent raised: {type(exc).__name__}: {exc}")
            return 1
        elapsed = time.time() - t0
        stamp(f"  agent finished in {elapsed:.1f}s")

        stamp("")
        stamp("=== TRAJECTORY ===")
        trajectory = getattr(pred, "trajectory", None)
        if not trajectory:
            print("  (empty / no trajectory)")
        else:
            print(f"  type: {type(trajectory).__name__}")
            print(f"  raw: {trajectory!r}"[:500])
            if isinstance(trajectory, dict):
                # dspy.ReAct may emit a dict like {"thought_0": ..., "tool_name_0": ..., ...}
                items = sorted(trajectory.items())
                for k, v in items:
                    val_str = str(v)
                    if len(val_str) > 300:
                        val_str = val_str[:300] + "..."
                    print(f"  {k}: {val_str}")
            elif isinstance(trajectory, list):
                for i, step in enumerate(trajectory):
                    print(f"  step {i}: {step!r}"[:500])
        stamp("")
        stamp("=== FINAL ANSWER ===")
        final_answer = getattr(pred, "final_answer", None)
        print(final_answer if final_answer else "(none)")
        stamp("")

        stamp("=== JUDGE VERDICT ===")
        judge = dspy.Predict(Assess)
        try:
            result = judge(
                user_request=ex.user_request,
                trajectory=str(trajectory or ""),
                final_answer=str(final_answer or ""),
            )
            stamp(f"  success: {result.success}")
            stamp(f"  reason: {result.reason}")
        except Exception as exc:
            stamp(f"  ✗ judge raised: {type(exc).__name__}: {exc}")
            return 1

    stamp("")
    stamp("=== DONE ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
