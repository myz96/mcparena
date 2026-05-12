"""Unit tests for pilot.py helpers that don't require live MCP / LLM access.

Covers: `_validate_server_id`, `_find_spec`, `_replicate_trials`, `_format_result`,
`_extract_exemplars`. The heavy `run_*` functions are exercised live by the
smoke / shake-out / full pipeline; hermetic tests for them would require
extensive MCP/LLM mocking that would test mocks more than logic.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from mcparena.pilot import pilot as pilot_module
from mcparena.pilot.tasks import PILOT_SERVERS


def test_validate_server_id_accepts_known_servers() -> None:
    for spec in PILOT_SERVERS:
        pilot_module._validate_server_id(spec.name)  # no raise


def test_validate_server_id_rejects_typos() -> None:
    with pytest.raises(ValueError, match="Unknown server_id 'math_mcp_typo'"):
        pilot_module._validate_server_id("math_mcp_typo")


def test_find_spec_returns_matching_serverspec() -> None:
    spec = pilot_module._find_spec("math_mcp")
    assert spec.name == "math_mcp"
    assert spec.mcp_bench_id == "Math MCP"


def test_find_spec_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="Unknown server"):
        pilot_module._find_spec("nonexistent")


def test_replicate_trials_expands_examples_n_times() -> None:
    examples = ["a", "b", "c"]
    trials = pilot_module._replicate_trials(examples, n_trials=4)
    assert len(trials) == 12
    assert trials.count("a") == 4
    assert trials.count("b") == 4
    assert trials.count("c") == 4


def test_replicate_trials_with_zero_trials_returns_empty() -> None:
    assert pilot_module._replicate_trials(["x"], n_trials=0) == []


def test_format_result_extracts_scores_from_evaluation_result() -> None:
    """Convert dspy.Evaluate's `EvaluationResult` shape to a JSON-serializable dict."""
    fake_eval = SimpleNamespace(
        score=0.75,
        results=[
            (SimpleNamespace(), SimpleNamespace(), 1.0),
            (SimpleNamespace(), SimpleNamespace(), 0.5),
            (SimpleNamespace(), SimpleNamespace(), 0.75),
        ],
    )
    formatted = pilot_module._format_result(fake_eval, "math_mcp", "baseline")
    assert formatted["server_id"] == "math_mcp"
    assert formatted["condition"] == "baseline"
    assert formatted["mean_score"] == 0.75
    assert formatted["n_trials"] == 3
    assert formatted["per_trial_scores"] == [1.0, 0.5, 0.75]


def test_format_result_falls_back_to_mean_when_no_aggregate_score() -> None:
    fake_eval = SimpleNamespace(
        results=[
            (SimpleNamespace(), SimpleNamespace(), 1.0),
            (SimpleNamespace(), SimpleNamespace(), 0.0),
        ]
    )
    formatted = pilot_module._format_result(fake_eval, "math_mcp", "baseline")
    assert formatted["mean_score"] == 0.5


def test_extract_exemplars_picks_first_success_per_tool() -> None:
    """`_extract_exemplars` walks `pred.trajectory` of successful trials and
    captures the FIRST winning tool-call per tool name."""

    def _pred(steps: list[dict[str, Any]]) -> Any:
        return SimpleNamespace(trajectory=steps)

    fake_eval = SimpleNamespace(
        results=[
            # Score 1.0 — successful — extract its tool calls
            (
                SimpleNamespace(),
                _pred(
                    [
                        {"selected_fn": "mean", "args": {"x": [1, 2, 3]}, "fn_output": "2.0"},
                        {"selected_fn": "sum", "args": {"x": [1, 2]}, "fn_output": "3"},
                    ]
                ),
                1.0,
            ),
            # Score 0.0 — failed — skip
            (
                SimpleNamespace(),
                _pred([{"selected_fn": "mean", "args": {"x": [99]}, "fn_output": "99"}]),
                0.0,
            ),
            # Score 1.0 again — but `mean` already captured; `stddev` is new
            (
                SimpleNamespace(),
                _pred(
                    [
                        {"selected_fn": "mean", "args": {"x": [10]}, "fn_output": "10"},
                        {"selected_fn": "stddev", "args": {"x": [1, 2, 3]}, "fn_output": "0.82"},
                    ]
                ),
                1.0,
            ),
        ]
    )
    exemplars = pilot_module._extract_exemplars(fake_eval)
    assert set(exemplars.keys()) == {"mean", "sum", "stddev"}
    # First success won — `mean` exemplar is from trial 1, not trial 3
    assert exemplars["mean"]["input"] == {"x": [1, 2, 3]}
    assert exemplars["mean"]["output"] == "2.0"


def test_extract_exemplars_empty_when_no_successes() -> None:
    fake_eval = SimpleNamespace(results=[(SimpleNamespace(), SimpleNamespace(trajectory=[]), 0.0)])
    assert pilot_module._extract_exemplars(fake_eval) == {}
