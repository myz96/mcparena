"""LLM-as-judge for the mcparena pilot.

The `Assess` Signature IS the published "judge prompt" — its docstring +
InputField/OutputField descriptions form the prompt the judge LM sees.
Snapshot-tested in `tests/test_judge.py` to catch unintentional drift.

Pilot uses same-model judging (judge = program LM = Sonnet 4 via OpenRouter).
Cross-model judging is Phase 1.

Two metric wrappers expose the judge through the two DSPy contracts:
- `judge_metric_evaluate(example, pred, trace=None) -> float`                            (3-arg, for dspy.Evaluate + dspy.MIPROv2)
- `judge_metric_gepa(gold, pred, trace, pred_name, pred_trace) -> dspy.Prediction`       (5-arg, for dspy.GEPA — returns Prediction(score, feedback))

The judge evaluates the agent's trajectory + final answer purely against the
user_request narrative (which, for MCP-Bench tasks, IS the success criteria —
their tasks are detailed procedural specs).
"""

from __future__ import annotations

from typing import Any

import dspy


class Assess(dspy.Signature):  # type: ignore[misc]  # dspy.Signature is Any-typed
    """Judge whether the agent's tool-use trajectory accomplished the user's task.

    Examine the user_request (a detailed procedural specification), the full
    trajectory (reasoning + tool calls + tool outputs), and the final answer.
    Return success=True only if the trajectory and final answer satisfy all
    steps and requirements stated in the user_request. Provide a one-sentence
    reason naming the specific success or failure mode.
    """

    user_request: str = dspy.InputField(
        desc="The user's task — a detailed procedural specification. Treat as the success criteria."
    )
    trajectory: str = dspy.InputField(
        desc="Full ReAct trajectory (reasoning + tool calls + outputs)"
    )
    final_answer: str = dspy.InputField()
    success: bool = dspy.OutputField()
    reason: str = dspy.OutputField()


def _judge_core(example: Any, pred: Any) -> tuple[float, str]:
    """Shared judge invocation used by both metric wrappers."""
    judge = dspy.Predict(Assess)
    result = judge(
        user_request=example.user_request,
        trajectory=str(getattr(pred, "trajectory", "")),
        final_answer=getattr(pred, "final_answer", ""),
    )
    return (1.0 if result.success else 0.0, result.reason)


def judge_metric_evaluate(example: Any, pred: Any, trace: Any = None) -> float:
    """3-arg metric for `dspy.Evaluate` and `dspy.MIPROv2`."""
    score, _ = _judge_core(example, pred)
    return score


def judge_metric_gepa(
    gold: Any,
    pred: Any,
    trace: Any = None,
    pred_name: Any = None,
    pred_trace: Any = None,
) -> Any:
    """Metric for `dspy.GEPA` — returns `dspy.Prediction(score, feedback)`.

    Trailing args default to None because DSPy calls this in two contexts:
    (a) reflection feedback (5-arg), (b) plain evaluation (3-arg). Without
    defaults, the 3-arg path raises TypeError and every trial scores 0.0.
    """
    score, reason = _judge_core(gold, pred)
    return dspy.Prediction(score=score, feedback=reason)
