"""Debug harness for run_axis_ii hang.

Runs axis_ii STANDALONE (no preceding baseline/miprov2/gepa) with
timestamped print statements at every step, so we can see exactly
where the hang occurs.

Usage:
    set -a && source .env && set +a
    MCPARENA_PILOT_MODEL=openrouter/google/gemini-2.0-flash-lite-001 \\
      uv run python scripts/debug_axis_ii.py
"""

from __future__ import annotations

import sys
import time

sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]


def stamp(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> int:
    stamp("=== axis_ii debug harness ===")
    stamp("STEP 1: import dspy + mcparena modules")
    import dspy  # noqa: F401

    from mcparena.pilot import costs, tools
    from mcparena.pilot.benchmark import parse_server_tasks
    from mcparena.pilot.judge import judge_metric_evaluate
    from mcparena.pilot.lm import get_lm
    from mcparena.pilot.tasks import PILOT_SERVERS

    stamp("STEP 2: configure dspy LM")
    program_lm = get_lm()
    dspy.configure(lm=program_lm)
    stamp(f"  LM ready: {program_lm}")

    stamp("STEP 3: pick math_mcp spec + load tasks")
    spec = next(s for s in PILOT_SERVERS if s.name == "math_mcp")
    examples = parse_server_tasks(spec.mcp_bench_id)
    stamp(f"  loaded {len(examples)} tasks")

    stamp("STEP 4: open persistent_session — first MCP subprocess spawn")
    with tools.persistent_session(spec.to_stdio_params()) as session:
        stamp(f"  session opened; got {len(session.tool_specs)} tools")

        stamp("STEP 5: build tool wrappers")
        tool_list = tools.make_tools(session)
        stamp(f"  {len(tool_list)} dspy.Tool wrappers built")

        stamp("STEP 6: generate up to 4 permutations")
        perms = tools.permute_tools(tool_list, max_permutations=4)
        stamp(f"  {len(perms)} permutations to try")

        for i, perm in enumerate(perms):
            stamp(f"STEP 7.{i}: build ReAct for permutation {i}")
            program = dspy.ReAct("user_request -> final_answer", tools=perm, max_iters=5)

            n_trials = 2
            replicated = [ex for _ in range(n_trials) for ex in examples]
            stamp(f"  trials list size: {len(replicated)}")

            stamp(f"STEP 8.{i}: build dspy.Evaluate (num_threads=8)")
            evaluator = dspy.Evaluate(
                devset=replicated,
                metric=judge_metric_evaluate,
                num_threads=8,
                failure_score=0.0,
            )

            stamp(f"STEP 9.{i}: CALL evaluator(program) — this is the suspected hang point")
            t0 = time.time()
            result = evaluator(program)
            elapsed = time.time() - t0
            stamp(
                f"  evaluator returned after {elapsed:.1f}s, score={getattr(result, 'score', 'no .score')}"
            )

    stamp("STEP 10: session closed cleanly")
    delta = costs.absorb_lm_history(program_lm, role="program", condition="axis_ii_debug")
    stamp(f"  total cost from this run: ${delta:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
