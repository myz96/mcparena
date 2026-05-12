"""mcparena pilot runner — orchestrates the 5-condition spike.

Live implementation of the pre-registered pilot. Each `run_*` function:
1. Loads MCP-Bench tasks for the target server via `parse_server_tasks`.
2. Builds a `dspy.ReAct` program with the MCP server's tools wrapped as
   `dspy.Tool` closures (via `tools.make_tools`).
3. (Where applicable) compiles the program with `dspy.MIPROv2` /
   `dspy.GEPA` / axis-specific wrapper.
4. Evaluates via `dspy.Evaluate` with `failure_score=0.0` and our
   `judge_metric_evaluate` metric.
5. Absorbs token-usage into `costs.CostState` and checks caps.

Cost discipline (Sonnet 4 via OpenRouter; $3/M in, $15/M out):
- baseline / axis_ii / axis_iii  : Sonnet trials only        (~$10-15)
- miprov2                        : MIPROv2 compile + re-eval (~$20-25)
- gepa                           : GEPA compile + reflection (~$30-40)
- hard cap                       : $300 (raises RuntimeError)
- reflection share gate          : >60% of total raises
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from mcparena.pilot import costs, tools
from mcparena.pilot.benchmark import parse_server_tasks
from mcparena.pilot.judge import judge_metric_evaluate, judge_metric_gepa
from mcparena.pilot.lm import DEFAULT_PILOT_MODEL, get_lm
from mcparena.pilot.tasks import PILOT_SERVERS, ServerSpec

RESULTS_DIR = Path("pilot-results")
CONDITION_ORDER = ["baseline", "miprov2", "gepa", "axis_ii", "axis_iii"]


# ---- validation / setup helpers ----


def _valid_server_ids() -> set[str]:
    return {s.name for s in PILOT_SERVERS}


def _validate_server_id(server_id: str) -> None:
    if server_id not in _valid_server_ids():
        raise ValueError(f"Unknown server_id {server_id!r}. Valid: {sorted(_valid_server_ids())}")


def _find_spec(server_id: str) -> ServerSpec:
    for spec in PILOT_SERVERS:
        if spec.name == server_id:
            return spec
    raise ValueError(f"Unknown server: {server_id}")


def _assert_clean_tree(allow_dirty: bool) -> None:
    """R8: full pilot refuses to run on a dirty working tree."""
    if allow_dirty:
        return
    try:
        status = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise RuntimeError(
            "Could not verify git working-tree cleanliness — refusing to run "
            "the full pilot. Re-run from inside the repo, or pass --allow-dirty."
        ) from exc
    if status:
        raise RuntimeError(
            "Working tree is dirty; refusing to run the full pilot (pre-reg "
            "honor system requires clean tree).\n"
            f"git status --porcelain:\n{status}\n"
            "Commit/stash changes, or pass --allow-dirty."
        )


def _load_examples(server_id: str) -> list[Any]:
    spec = _find_spec(server_id)
    examples = parse_server_tasks(spec.mcp_bench_id)
    if not examples:
        raise RuntimeError(
            f"No tasks loaded for {server_id!r} ({spec.mcp_bench_id!r}). "
            "Run `python -c 'from mcparena.pilot.benchmark import "
            "ensure_mcp_bench_cloned; ensure_mcp_bench_cloned()'` first, or "
            "check that the pinned MCP-Bench SHA contains this server."
        )
    return examples


def _replicate_trials(examples: list[Any], n_trials: int) -> list[Any]:
    """Repeat each example n_trials times so dspy.Evaluate runs multiple trials."""
    return [ex for _ in range(n_trials) for ex in examples]


def _build_react(tool_list: list[Any]) -> Any:
    """Build a `dspy.ReAct` program with the given tools."""
    import dspy

    return dspy.ReAct("user_request -> final_answer", tools=tool_list, max_iters=5)


def _evaluate(program: Any, examples: list[Any]) -> Any:
    """Run `dspy.Evaluate` over the examples with `failure_score=0.0`."""
    import dspy

    evaluator = dspy.Evaluate(
        devset=examples,
        metric=judge_metric_evaluate,
        num_threads=8,
        failure_score=0.0,
    )
    return evaluator(program)


def _format_result(eval_result: Any, server_id: str, condition: str) -> dict[str, Any]:
    """Convert `EvaluationResult` to a JSON-serializable summary."""
    if hasattr(eval_result, "results"):
        scores = [float(score) for _, _, score in eval_result.results]
    else:
        scores = []
    mean = (
        float(eval_result.score)
        if hasattr(eval_result, "score") and eval_result.score is not None
        else (float(np.mean(scores)) if scores else 0.0)
    )
    return {
        "server_id": server_id,
        "condition": condition,
        "mean_score": mean,
        "n_trials": len(scores),
        "per_trial_scores": scores,
    }


# ---- individual conditions ----


def run_baseline(server_id: str, n_trials: int = 5) -> dict[str, Any]:
    """Vanilla `dspy.ReAct` against the server's MCP-Bench tasks."""
    _validate_server_id(server_id)
    spec = _find_spec(server_id)

    import dspy

    program_lm = get_lm()
    dspy.configure(lm=program_lm)

    tool_list = tools.make_tools(spec.to_stdio_params())
    program = _build_react(tool_list)

    examples = _load_examples(server_id)
    trials = _replicate_trials(examples, n_trials)
    result = _evaluate(program, trials)

    costs.absorb_lm_history(program_lm, role="program", condition="baseline")
    costs.check_cost_caps()
    return _format_result(result, server_id, "baseline")


def run_miprov2(server_id: str, n_trials: int = 5) -> dict[str, Any]:
    """`dspy.MIPROv2(auto="light")` over the program signature."""
    _validate_server_id(server_id)
    spec = _find_spec(server_id)

    import dspy

    program_lm = get_lm()
    dspy.configure(lm=program_lm)

    tool_list = tools.make_tools(spec.to_stdio_params())
    program = _build_react(tool_list)
    examples = _load_examples(server_id)

    optimizer = dspy.MIPROv2(metric=judge_metric_evaluate, auto="light", num_threads=8)
    optimized = optimizer.compile(program, trainset=examples)

    trials = _replicate_trials(examples, n_trials)
    result = _evaluate(optimized, trials)

    costs.absorb_lm_history(program_lm, role="program", condition="miprov2")
    costs.check_cost_caps()
    return _format_result(result, server_id, "miprov2")


def run_gepa(server_id: str, n_trials: int = 5) -> dict[str, Any]:
    """`dspy.GEPA(auto="light")` with same-model reflection LM."""
    _validate_server_id(server_id)
    spec = _find_spec(server_id)

    import dspy

    program_lm = get_lm()
    reflection_lm = get_lm(role="reflection")
    dspy.configure(lm=program_lm)

    tool_list = tools.make_tools(spec.to_stdio_params())
    program = _build_react(tool_list)
    examples = _load_examples(server_id)

    optimizer = dspy.GEPA(
        metric=judge_metric_gepa,
        reflection_lm=reflection_lm,
        auto="light",
        track_stats=True,
    )
    optimized = optimizer.compile(program, trainset=examples, valset=examples)

    trials = _replicate_trials(examples, n_trials)
    result = _evaluate(optimized, trials)

    costs.absorb_lm_history(reflection_lm, role="reflection", condition="gepa")
    costs.absorb_lm_history(program_lm, role="program", condition="gepa")
    costs.check_cost_caps()
    return _format_result(result, server_id, "gepa")


def run_axis_ii(server_id: str, n_trials: int = 5) -> dict[str, Any]:
    """Tool-ordering permutation search — try ≤4 permutations, keep the best."""
    _validate_server_id(server_id)
    spec = _find_spec(server_id)

    import dspy

    program_lm = get_lm()
    dspy.configure(lm=program_lm)

    tool_list = tools.make_tools(spec.to_stdio_params())
    examples = _load_examples(server_id)

    perms = tools.permute_tools(tool_list, max_permutations=4)
    per_perm_scores: list[float] = []
    best_result = None
    best_score = -1.0
    for perm in perms:
        program = _build_react(perm)
        trials = _replicate_trials(examples, n_trials)
        result = _evaluate(program, trials)
        score = float(getattr(result, "score", 0.0) or 0.0)
        per_perm_scores.append(score)
        if score > best_score:
            best_score = score
            best_result = result

    costs.absorb_lm_history(program_lm, role="program", condition="axis_ii")
    costs.check_cost_caps()
    formatted = _format_result(best_result, server_id, "axis_ii")
    formatted["per_permutation_scores"] = per_perm_scores
    return formatted


def run_axis_iii(server_id: str, n_trials: int = 5) -> dict[str, Any]:
    """1-shot exemplar injection — extract from baseline, inject into tool descriptions."""
    _validate_server_id(server_id)
    spec = _find_spec(server_id)

    import dspy

    program_lm = get_lm()
    dspy.configure(lm=program_lm)

    tool_list = tools.make_tools(spec.to_stdio_params())
    examples = _load_examples(server_id)

    baseline_program = _build_react(tool_list)
    exemplar_result = _evaluate(baseline_program, examples)
    exemplars = _extract_exemplars(exemplar_result)

    injected_tools = tools.inject_one_shot(tool_list, exemplars)
    injected_program = _build_react(injected_tools)
    trials = _replicate_trials(examples, n_trials)
    result = _evaluate(injected_program, trials)

    costs.absorb_lm_history(program_lm, role="program", condition="axis_iii")
    costs.check_cost_caps()
    formatted = _format_result(result, server_id, "axis_iii")
    formatted["exemplars_injected"] = list(exemplars.keys())
    return formatted


def _extract_exemplars(eval_result: Any) -> dict[str, dict[str, Any]]:
    """Pluck the first successful tool call per tool name from a baseline run."""
    exemplars: dict[str, dict[str, Any]] = {}
    if not hasattr(eval_result, "results"):
        return exemplars
    for _ex, pred, score in eval_result.results:
        if float(score) < 1.0:
            continue
        trajectory = getattr(pred, "trajectory", None)
        if not trajectory:
            continue
        for step in trajectory:
            if not isinstance(step, dict):
                continue
            name = step.get("selected_fn")
            args = step.get("args")
            output = step.get("fn_output") or step.get("result")
            if name and name not in exemplars and args is not None:
                exemplars[name] = {"input": args, "output": output or ""}
    return exemplars


# ---- smoke / shake-out / full ----


def run_smoke_adapter() -> int:
    """R9 gate (~$0): construct GEPA's MCP adapter against Math MCP.

    Verifies (a) the MCP server actually launches and lists tools, and (b)
    `gepa.adapters.mcp_adapter.MCPAdapter` can be constructed with our config.
    No LLM calls. If this fails on a fresh setup, run MCP-Bench's
    `mcp_servers/install.sh` first.
    """
    from gepa.adapters.mcp_adapter import MCPAdapter

    spec = _find_spec("math_mcp")
    print(f"smoke-adapter: connecting to {spec.name} via {spec.command} {spec.args}")
    print(f"  cwd={spec.cwd}")

    try:
        tool_specs = tools.discover_tool_specs(spec.to_stdio_params())
    except Exception as exc:
        print(f"✗ Could not connect: {type(exc).__name__}: {exc}")
        print("  Did you run 'bash third_party/mcp-bench-tasks/mcp_servers/install.sh'?")
        return 1
    tool_names = [t["name"] for t in tool_specs]
    print(f"✓ Discovered {len(tool_names)} tools: {tool_names}")

    try:
        MCPAdapter(
            tool_names=tool_names,
            task_model=DEFAULT_PILOT_MODEL,
            metric_fn=lambda _item, _output: 1.0,
            server_params=spec.to_stdio_params(),
        )
    except Exception as exc:
        print(f"✗ MCPAdapter construction failed: {type(exc).__name__}: {exc}")
        return 1
    print("✓ MCPAdapter constructed (R9 gate passed)")
    return 0


def run_smoke_budget(server_id: str = "math_mcp", n_trials: int = 2) -> int:
    """Smoke (~$8): 1 server × all tasks × n_trials trials × 5 conditions."""
    _validate_server_id(server_id)
    print(f"smoke-budget: {server_id} × {n_trials} trials × 5 conditions")

    results = _run_all_conditions(server_id, n_trials=n_trials)
    aggregate_and_report({server_id: results}, mode="smoke-budget")
    _print_cost_summary()
    return 0


def run_shake_out(server_id: str = "math_mcp", n_trials: int = 3) -> int:
    """Shake-out (~$30): 1 server × all tasks × n_trials × 5 conditions."""
    _validate_server_id(server_id)
    print(f"shake-out: {server_id} × {n_trials} trials × 5 conditions")

    results = _run_all_conditions(server_id, n_trials=n_trials)
    aggregate_and_report({server_id: results}, mode="shake-out")
    _print_cost_summary()
    return 0


def _run_all_conditions(server_id: str, n_trials: int) -> dict[str, dict[str, Any]]:
    """Run all 5 conditions for one server, collecting per-condition results."""
    fns = {
        "baseline": run_baseline,
        "miprov2": run_miprov2,
        "gepa": run_gepa,
        "axis_ii": run_axis_ii,
        "axis_iii": run_axis_iii,
    }
    results: dict[str, dict[str, Any]] = {}
    for cond in CONDITION_ORDER:
        print(f"  → {cond}")
        try:
            results[cond] = fns[cond](server_id, n_trials=n_trials)
        except Exception as exc:
            print(f"  ✗ {cond} failed: {type(exc).__name__}: {exc}")
            results[cond] = {
                "server_id": server_id,
                "condition": cond,
                "error": str(exc),
                "per_trial_scores": [],
                "mean_score": 0.0,
                "n_trials": 0,
            }
    return results


def _print_cost_summary() -> None:
    state = costs.get_state()
    print(f"\ncost summary: total ${state.total_usd:.4f}")
    print(f"  program:    ${state.program_usd:.4f}")
    print(f"  reflection: ${state.reflection_usd:.4f}")
    if state.by_condition:
        print(f"  by condition: {dict(state.by_condition)}")


# ---- aggregation ----


def aggregate_and_report(
    results: dict[str, dict[str, dict[str, Any]]],
    mode: str = "full",
) -> None:
    """Compute paired 95% bootstrap CI per (server, condition) vs baseline.

    Writes ``pilot-results/<mode>.json`` with deltas + CIs + cost breakdown.
    The pilot memo template (Phase 1.5) reads this for narrative population.
    """
    import scipy.stats as stats

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {"mode": mode, "servers": {}}
    for server_id, conditions in results.items():
        if "baseline" not in conditions:
            summary["servers"][server_id] = {"error": "missing baseline"}
            continue
        baseline_scores = np.array(conditions["baseline"].get("per_trial_scores", []))
        server_data: dict[str, Any] = {
            "baseline_mean": float(baseline_scores.mean()) if baseline_scores.size else 0.0,
            "n_baseline_trials": int(baseline_scores.size),
            "deltas": {},
        }
        for cond in CONDITION_ORDER:
            if cond == "baseline" or cond not in conditions:
                continue
            scores = np.array(conditions[cond].get("per_trial_scores", []))
            n_paired = int(min(scores.size, baseline_scores.size))
            if n_paired == 0:
                server_data["deltas"][cond] = {"error": "no samples"}
                continue
            b_sample = baseline_scores[:n_paired]
            o_sample = scores[:n_paired]
            try:
                ci = stats.bootstrap(
                    (o_sample, b_sample),
                    statistic=lambda o, b: float(np.mean(o) - np.mean(b)),
                    n_resamples=1000,
                    confidence_level=0.95,
                    paired=True,
                    method="percentile",
                )
                server_data["deltas"][cond] = {
                    "delta_pp": float(np.mean(o_sample) - np.mean(b_sample)) * 100,
                    "ci_low_pp": float(ci.confidence_interval.low) * 100,
                    "ci_high_pp": float(ci.confidence_interval.high) * 100,
                    "n_paired": n_paired,
                    "condition_mean": float(np.mean(o_sample)),
                }
            except Exception as exc:
                server_data["deltas"][cond] = {"error": str(exc), "n_paired": n_paired}
        summary["servers"][server_id] = server_data

    state = costs.get_state()
    summary["cost"] = {
        "total_usd": round(state.total_usd, 4),
        "program_usd": round(state.program_usd, 4),
        "reflection_usd": round(state.reflection_usd, 4),
        "reflection_share": round(state.reflection_share, 4),
        "by_condition": {k: round(v, 4) for k, v in state.by_condition.items()},
    }
    summary["raw_results"] = {
        srv: {cond: data for cond, data in conds.items()} for srv, conds in results.items()
    }

    out = RESULTS_DIR / f"{mode}.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\n✓ Wrote {out}")


# ---- main ----


def main(args: argparse.Namespace) -> int:
    """CLI entrypoint dispatched from `mcparena.cli.main`."""
    if args.server is not None:
        _validate_server_id(args.server)

    costs.reset()

    if args.smoke_adapter:
        return run_smoke_adapter()
    if args.smoke_budget:
        return run_smoke_budget(server_id=args.server or "math_mcp")
    if args.shake_out:
        return run_shake_out(server_id=args.server or "math_mcp")

    _assert_clean_tree(allow_dirty=args.allow_dirty)

    servers_to_run = [args.server] if args.server else [s.name for s in PILOT_SERVERS]
    conditions_filter = args.condition

    full_results: dict[str, dict[str, dict[str, Any]]] = {}
    for server_id in servers_to_run:
        print(f"\n=== {server_id} ===")
        if conditions_filter == "all":
            server_results = _run_all_conditions(server_id, n_trials=5)
        else:
            fns = {
                "baseline": run_baseline,
                "miprov2": run_miprov2,
                "gepa": run_gepa,
                "axis_ii": run_axis_ii,
                "axis_iii": run_axis_iii,
            }
            print(f"  → {conditions_filter}")
            server_results = {conditions_filter: fns[conditions_filter](server_id, n_trials=5)}
        full_results[server_id] = server_results

    aggregate_and_report(full_results, mode="full")
    _print_cost_summary()
    return 0
