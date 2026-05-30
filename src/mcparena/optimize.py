"""`mcparena optimize` — the distribution wedge.

Point at any MCP server command, supply a JSON list of tasks, get back
the GEPA-discovered constraints (the wrong-kwarg quirks the base model
hallucinates) plus a baseline-vs-GEPA delta and bootstrap CI.

For pre-registered pilot servers, prefer `mcparena pilot` — it pins the
MCP-Bench task set and runs all 5 conditions. `optimize` is the
generalized variant for arbitrary servers.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from mcp.client.stdio import StdioServerParameters

from mcparena.pilot import costs, tools
from mcparena.pilot.judge import judge_metric_evaluate, judge_metric_gepa
from mcparena.pilot.lm import get_lm


@dataclass
class OptimizeConfig:
    server_cmd: str
    server_args: list[str]
    tasks: list[dict[str, Any]]
    n_trials: int = 3
    output_dir: Path = Path("mcparena-optimize-results")
    max_full_evals: int = 1
    server_env: dict[str, str] | None = None
    server_cwd: str | None = None

    def to_stdio_params(self) -> StdioServerParameters:
        return StdioServerParameters(
            command=self.server_cmd,
            args=self.server_args,
            env=self.server_env or {},
            cwd=self.server_cwd,
        )


def _build_examples(tasks: list[dict[str, Any]]) -> list[Any]:
    import dspy

    examples: list[Any] = []
    for i, t in enumerate(tasks):
        if "user_request" not in t:
            raise ValueError(f"tasks[{i}] missing required field 'user_request'")
        examples.append(
            dspy.Example(
                task_id=t.get("task_id", f"task_{i}"), user_request=t["user_request"]
            ).with_inputs("user_request")
        )
    return examples


def _replicate(examples: list[Any], n_trials: int) -> list[Any]:
    return [ex for _ in range(n_trials) for ex in examples]


def _build_react(tool_list: list[Any]) -> Any:
    import dspy

    return dspy.ReAct("user_request -> final_answer", tools=tool_list, max_iters=20)


def _evaluate(program: Any, examples: list[Any]) -> Any:
    import dspy

    return dspy.Evaluate(
        devset=examples, metric=judge_metric_evaluate, num_threads=8, failure_score=0.0
    )(program)


def _format_eval(eval_result: Any, condition: str) -> dict[str, Any]:
    scores = (
        [float(s) for _, _, s in eval_result.results] if hasattr(eval_result, "results") else []
    )
    return {
        "condition": condition,
        "mean_score": float(np.mean(scores)) if scores else 0.0,
        "n_trials": len(scores),
        "per_trial_scores": scores,
    }


def _extract_discovered_prompt(optimized_program: Any) -> str | None:
    """Pull the GEPA-discovered ReAct instructions from the compiled program.

    DSPy stores the optimized signature on the Predict module. The instructions
    string is what GEPA rewrote — typically including a `### Critical Task
    Constraints` block listing tool-schema quirks the optimizer found.
    """
    react = getattr(optimized_program, "react", None) or getattr(optimized_program, "predict", None)
    if react is None:
        return None
    signature = getattr(react, "signature", None)
    if signature is None:
        return None
    return getattr(signature, "instructions", None)


def _bootstrap_delta(baseline_scores: list[float], gepa_scores: list[float]) -> dict[str, Any]:
    import scipy.stats as stats

    n = min(len(baseline_scores), len(gepa_scores))
    if n == 0:
        return {"error": "no paired samples"}
    b = np.array(baseline_scores[:n])
    g = np.array(gepa_scores[:n])
    delta_pp = float(np.mean(g) - np.mean(b)) * 100
    # scipy.stats.bootstrap requires ≥2 observations per sample. With n=1
    # there is no CI to compute — return the point estimate only.
    if n < 2:
        return {
            "delta_pp": delta_pp,
            "n_paired": n,
            "ci_skipped": "n_paired < 2; bootstrap requires ≥2 observations",
        }
    ci = stats.bootstrap(
        (g, b),
        statistic=lambda gg, bb: float(np.mean(gg) - np.mean(bb)),
        n_resamples=1000,
        confidence_level=0.95,
        paired=True,
        method="percentile",
    )
    return {
        "delta_pp": delta_pp,
        "ci_low_pp": float(ci.confidence_interval.low) * 100,
        "ci_high_pp": float(ci.confidence_interval.high) * 100,
        "n_paired": n,
    }


def _write_summary_markdown(
    out: Path,
    baseline: dict[str, Any],
    gepa: dict[str, Any],
    delta: dict[str, Any],
    discovered_prompt: str | None,
    cost_usd: float,
) -> None:
    delta_pp = delta.get("delta_pp", 0.0)
    lines = [
        "# mcparena optimize — results",
        "",
        "| Condition | Mean | n | ",
        "|---|---|---|",
        f"| baseline (`dspy.ReAct`) | {baseline['mean_score']:.1%} | {baseline['n_trials']} |",
        f"| **gepa** | **{gepa['mean_score']:.1%}** | {gepa['n_trials']} |",
        "",
        f"**Δ:** {delta_pp:+.2f}pp   ",
    ]
    if "ci_low_pp" in delta and "ci_high_pp" in delta:
        lines.append(f"**95% CI:** [{delta['ci_low_pp']:+.2f}pp, {delta['ci_high_pp']:+.2f}pp]   ")
    elif "ci_skipped" in delta:
        lines.append(f"**95% CI:** *skipped — {delta['ci_skipped']}*   ")
    lines += [f"**cost:** ${cost_usd:.4f}", ""]
    if discovered_prompt:
        lines += [
            "## What GEPA discovered about your MCP server",
            "",
            "Below is the prompt GEPA evolved by reflecting on baseline failures.",
            "If it includes a `Critical Task Constraints` (or similar) section,",
            "those are the tool-schema quirks the base model couldn't infer from",
            "your tool descriptions — fix them upstream and the lift may shrink.",
            "",
            "```",
            discovered_prompt,
            "```",
            "",
        ]
    out.write_text("\n".join(lines))


def run_optimize(config: OptimizeConfig) -> dict[str, Any]:
    """Run baseline + GEPA against an arbitrary MCP server and write results."""
    import dspy

    config.output_dir.mkdir(parents=True, exist_ok=True)
    costs.reset()

    program_lm = get_lm()
    reflection_lm = get_lm(role="reflection")
    dspy.configure(lm=program_lm)
    examples = _build_examples(config.tasks)

    print(
        f"[{time.strftime('%H:%M:%S')}] optimize: {len(examples)} tasks × "
        f"n_trials={config.n_trials} per condition",
        flush=True,
    )

    stdio_params = config.to_stdio_params()
    with tools.persistent_session(stdio_params) as session:
        tool_list = tools.make_tools(session)

        print(f"[{time.strftime('%H:%M:%S')}] → baseline", flush=True)
        baseline_program = _build_react(tool_list)
        baseline_eval = _evaluate(baseline_program, _replicate(examples, config.n_trials))
        baseline_result = _format_eval(baseline_eval, "baseline")
        costs.absorb_lm_history(program_lm, role="program", condition="baseline")
        print(
            f"[{time.strftime('%H:%M:%S')}] baseline mean: "
            f"{baseline_result['mean_score']:.1%} (n={baseline_result['n_trials']})",
            flush=True,
        )

        print(
            f"[{time.strftime('%H:%M:%S')}] → gepa (max_full_evals={config.max_full_evals})",
            flush=True,
        )
        gepa_program = _build_react(tool_list)
        gepa_optimizer = dspy.GEPA(
            metric=judge_metric_gepa,
            reflection_lm=reflection_lm,
            max_full_evals=config.max_full_evals,
            track_stats=True,
        )
        optimized = gepa_optimizer.compile(gepa_program, trainset=examples, valset=examples)
        gepa_eval = _evaluate(optimized, _replicate(examples, config.n_trials))
        gepa_result = _format_eval(gepa_eval, "gepa")
        costs.absorb_lm_history(reflection_lm, role="reflection", condition="gepa")
        costs.absorb_lm_history(program_lm, role="program", condition="gepa")
        print(
            f"[{time.strftime('%H:%M:%S')}] gepa mean: "
            f"{gepa_result['mean_score']:.1%} (n={gepa_result['n_trials']})",
            flush=True,
        )

    discovered_prompt = _extract_discovered_prompt(optimized)
    delta = _bootstrap_delta(baseline_result["per_trial_scores"], gepa_result["per_trial_scores"])
    cost_state = costs.get_state()

    summary = {
        "config": {
            "server_cmd": config.server_cmd,
            "server_args": config.server_args,
            "n_trials": config.n_trials,
            "max_full_evals": config.max_full_evals,
            "n_tasks": len(examples),
        },
        "baseline": baseline_result,
        "gepa": gepa_result,
        "delta": delta,
        "discovered_prompt": discovered_prompt,
        "cost": {
            "total_usd": round(cost_state.total_usd, 4),
            "program_usd": round(cost_state.program_usd, 4),
            "reflection_usd": round(cost_state.reflection_usd, 4),
        },
    }

    (config.output_dir / "results.json").write_text(json.dumps(summary, indent=2))
    _write_summary_markdown(
        config.output_dir / "summary.md",
        baseline_result,
        gepa_result,
        delta,
        discovered_prompt,
        cost_state.total_usd,
    )

    print(
        f"\n[{time.strftime('%H:%M:%S')}] ✓ wrote {config.output_dir}/results.json + summary.md",
        flush=True,
    )
    return summary


def load_config_from_args(
    server_cmd: str,
    server_args: list[str],
    tasks_path: Path,
    n_trials: int,
    output_dir: Path,
    max_full_evals: int,
    server_cwd: str | None,
) -> OptimizeConfig:
    """Build an OptimizeConfig from CLI inputs."""
    raw = json.loads(tasks_path.read_text())
    if not isinstance(raw, list):
        raise ValueError(f"{tasks_path} must contain a JSON list of task objects")
    return OptimizeConfig(
        server_cmd=server_cmd,
        server_args=server_args,
        tasks=raw,
        n_trials=n_trials,
        output_dir=output_dir,
        max_full_evals=max_full_evals,
        server_cwd=server_cwd,
    )
