"""Unit tests for judge_metric_evaluate / judge_metric_gepa.

These wrappers are called every trial by `dspy.Evaluate`, `dspy.MIPROv2`, and
`dspy.GEPA`. A bug (wrong return type, missing fields on the GEPA Prediction,
swallowed exception path) corrupts every pilot result. Hermetic — uses a
stub `dspy.Predict(Assess)` via monkeypatch, no live LLM calls.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import dspy
import pytest

from mcparena.pilot.judge import (
    Assess,
    _judge_core,
    judge_metric_evaluate,
    judge_metric_gepa,
)


@pytest.fixture
def stub_predict(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Replace `dspy.Predict(Assess)` with a stub returning configurable result."""
    stub_result = SimpleNamespace(success=True, reason="stubbed success")

    def fake_predict(signature: Any) -> Any:
        assert signature is Assess, "judge must call dspy.Predict(Assess)"
        return lambda **_kwargs: stub_result

    monkeypatch.setattr(dspy, "Predict", fake_predict)
    return {"result": stub_result}


def _example_and_pred() -> tuple[Any, Any]:
    example = SimpleNamespace(user_request="test task")
    pred = SimpleNamespace(trajectory="step 1 → step 2", final_answer="done")
    return example, pred


def test_judge_metric_evaluate_returns_float_one_for_success(stub_predict: dict[str, Any]) -> None:
    ex, pred = _example_and_pred()
    score = judge_metric_evaluate(ex, pred)
    assert score == 1.0
    assert isinstance(score, float)


def test_judge_metric_evaluate_returns_float_zero_for_failure(
    stub_predict: dict[str, Any],
) -> None:
    stub_predict["result"].success = False
    stub_predict["result"].reason = "stubbed failure"
    ex, pred = _example_and_pred()
    assert judge_metric_evaluate(ex, pred) == 0.0


def test_judge_metric_evaluate_accepts_trace_kwarg(stub_predict: dict[str, Any]) -> None:
    """dspy.MIPROv2 passes `trace` when bootstrapping demos — must accept it."""
    ex, pred = _example_and_pred()
    score = judge_metric_evaluate(ex, pred, trace=[("step", "result")])
    assert score == 1.0


def test_judge_metric_gepa_returns_prediction_with_score_and_feedback(
    stub_predict: dict[str, Any],
) -> None:
    ex, pred = _example_and_pred()
    result = judge_metric_gepa(ex, pred, trace=None, pred_name="program", pred_trace=None)
    assert isinstance(result, dspy.Prediction)
    assert result.score == 1.0
    assert result.feedback == "stubbed success"


def test_judge_metric_gepa_propagates_failure(stub_predict: dict[str, Any]) -> None:
    stub_predict["result"].success = False
    stub_predict["result"].reason = "wrong tool"
    ex, pred = _example_and_pred()
    result = judge_metric_gepa(ex, pred, trace=None, pred_name="p", pred_trace=None)
    assert result.score == 0.0
    assert result.feedback == "wrong tool"


def test_judge_core_handles_pred_without_trajectory(stub_predict: dict[str, Any]) -> None:
    """If a pred lacks `.trajectory` (e.g., a non-ReAct module), default to ''."""
    example = SimpleNamespace(user_request="t")
    pred = SimpleNamespace(final_answer="x")  # no .trajectory
    score, reason = _judge_core(example, pred)
    assert score == 1.0
    assert reason == "stubbed success"
