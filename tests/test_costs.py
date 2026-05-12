"""Unit tests for `mcparena.pilot.costs`.

Hermetic — no LLM calls. Tests the cost math, the cap enforcement, the
history absorption logic, and the reset semantics.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from mcparena.pilot import costs


@pytest.fixture(autouse=True)
def _reset_costs() -> None:
    """Each test gets a clean global state."""
    costs.reset()


def test_initial_state_is_zero() -> None:
    state = costs.get_state()
    assert state.total_usd == 0.0
    assert state.program_usd == 0.0
    assert state.reflection_usd == 0.0
    assert state.reflection_share == 0.0


def test_add_program_cost_only() -> None:
    state = costs.get_state()
    # 1M input tokens at $3 + 1M output tokens at $15 = $18 total
    cost = state.add(1_000_000, 1_000_000, role="program", condition="baseline")
    assert cost == pytest.approx(18.0)
    assert state.program_usd == pytest.approx(18.0)
    assert state.reflection_usd == 0.0
    assert state.reflection_share == 0.0


def test_add_reflection_cost_attributes_correctly() -> None:
    state = costs.get_state()
    state.add(100_000, 100_000, role="program", condition="gepa")
    state.add(100_000, 100_000, role="reflection", condition="gepa")
    # Each adds 0.3 + 1.5 = 1.8 → totals: program 1.8, reflection 1.8, total 3.6
    assert state.program_usd == pytest.approx(1.8)
    assert state.reflection_usd == pytest.approx(1.8)
    assert state.total_usd == pytest.approx(3.6)
    assert state.reflection_share == pytest.approx(0.5)


def test_by_condition_breakdown() -> None:
    state = costs.get_state()
    state.add(1_000_000, 0, role="program", condition="baseline")  # $3
    state.add(1_000_000, 0, role="program", condition="miprov2")  # $3
    state.add(1_000_000, 0, role="program", condition="baseline")  # +$3 = $6
    assert state.by_condition["baseline"] == pytest.approx(6.0)
    assert state.by_condition["miprov2"] == pytest.approx(3.0)


def test_check_caps_under_limit_passes() -> None:
    costs.get_state().add(100, 100, role="program", condition="baseline")
    costs.check_cost_caps()  # no raise


def test_check_caps_raises_over_hard_cap() -> None:
    # 100M output × $15/M = $1500 > $300 cap
    costs.get_state().add(0, 100_000_000, role="program", condition="baseline")
    with pytest.raises(RuntimeError, match="Cost cap exceeded"):
        costs.check_cost_caps()


def test_check_caps_raises_over_reflection_share() -> None:
    state = costs.get_state()
    # 10% program, 90% reflection
    state.add(1_000, 1_000, role="program", condition="gepa")
    state.add(9_000, 9_000, role="reflection", condition="gepa")
    with pytest.raises(RuntimeError, match="Reflection share"):
        costs.check_cost_caps()


def test_absorb_lm_history_drains_and_sums() -> None:
    """`absorb_lm_history` should sum tokens from `lm.history`, attribute to role/condition,
    and clear the history list (so subsequent calls don't double-count)."""
    fake_lm = SimpleNamespace(
        history=[
            {"usage": {"prompt_tokens": 1_000, "completion_tokens": 500}},
            {"usage": {"prompt_tokens": 2_000, "completion_tokens": 1_000}},
        ]
    )
    delta = costs.absorb_lm_history(fake_lm, role="program", condition="baseline")
    # 3000 input × $3/M + 1500 output × $15/M = 0.009 + 0.0225 = 0.0315
    assert delta == pytest.approx(0.0315)
    assert fake_lm.history == []  # drained
    # Subsequent absorb on same (now-empty) history adds nothing
    delta2 = costs.absorb_lm_history(fake_lm, role="program", condition="baseline")
    assert delta2 == 0.0


def test_absorb_lm_history_handles_missing_history() -> None:
    """An LM without a `.history` attribute returns 0.0 cleanly."""
    fake_lm: Any = SimpleNamespace()  # no .history
    assert costs.absorb_lm_history(fake_lm, role="program", condition="baseline") == 0.0


def test_absorb_lm_history_handles_nested_usage_shapes() -> None:
    """DSPy's history entries may nest `usage` under `response`/`raw`."""
    fake_lm = SimpleNamespace(
        history=[
            {"response": {"usage": {"prompt_tokens": 1_000, "completion_tokens": 1_000}}},
        ]
    )
    delta = costs.absorb_lm_history(fake_lm, role="program", condition="baseline")
    # 1000 × $3/M + 1000 × $15/M = 0.003 + 0.015 = 0.018
    assert delta == pytest.approx(0.018)


def test_reset_zeros_state_across_invocations() -> None:
    state = costs.get_state()
    state.add(100_000, 100_000, role="program", condition="baseline")
    assert costs.get_state().total_usd > 0
    costs.reset()
    assert costs.get_state().total_usd == 0.0
    assert costs.get_state().by_condition == {}
