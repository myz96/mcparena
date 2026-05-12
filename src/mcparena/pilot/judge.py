"""LLM-as-judge for the mcparena pilot.

The `Assess` Signature IS the published "judge prompt" — its docstring +
InputField/OutputField descriptions form the prompt the judge LM sees.
Snapshot-tested in `tests/test_judge.py` to catch unintentional drift.

Pilot uses same-model judging (judge = program LM = Sonnet 4.6). Cross-model
judging is Phase 1.

Two metric wrappers expose the judge through the two DSPy contracts:
- `judge_metric_evaluate(example, pred, trace=None) -> float`        (3-arg, for dspy.Evaluate + dspy.MIPROv2)
- `judge_metric_gepa(gold, pred, trace, pred_name, pred_trace)`      (5-arg, for dspy.GEPA — returns dspy.Prediction(score, feedback))
"""

from __future__ import annotations

from typing import Any

import dspy


class Assess(dspy.Signature):  # type: ignore[misc]  # dspy.Signature is Any-typed
    """Judge whether the agent's tool-use trajectory accomplished the user's task.

    Examine the user_request, the full trajectory (reasoning + tool calls + tool
    outputs), and the final answer. Return success=True only if the user's task
    was completed correctly via appropriate tool use. Provide a one-sentence
    reason that names the specific success or failure mode.
    """

    user_request: str = dspy.InputField()
    trajectory: str = dspy.InputField(
        desc="Full ReAct trajectory (reasoning + tool calls + outputs)"
    )
    final_answer: str = dspy.InputField()
    expected_outcome: str = dspy.InputField(desc="What success looks like for this task")
    success: bool = dspy.OutputField()
    reason: str = dspy.OutputField()


def _judge_core(example: Any, pred: Any) -> tuple[float, str]:
    """Shared judge invocation used by both metric wrappers."""
    judge = dspy.Predict(Assess)
    result = judge(
        user_request=example.user_request,
        trajectory=str(getattr(pred, "trajectory", "")),
        final_answer=getattr(pred, "final_answer", ""),
        expected_outcome=example.expected_outcome,
    )
    return (1.0 if result.success else 0.0, result.reason)


def judge_metric_evaluate(example: Any, pred: Any, trace: Any = None) -> float:
    """3-arg metric for `dspy.Evaluate` and `dspy.MIPROv2`."""
    score, _ = _judge_core(example, pred)
    return score


def judge_metric_gepa(
    gold: Any,
    pred: Any,
    trace: Any,
    pred_name: Any,
    pred_trace: Any,
) -> Any:
    """5-arg metric for `dspy.GEPA` — returns `dspy.Prediction(score, feedback)`."""
    score, reason = _judge_core(gold, pred)
    return dspy.Prediction(score=score, feedback=reason)
