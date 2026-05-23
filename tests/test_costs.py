"""Unit tests for `mcparena.pilot.costs` — pricing math + cap enforcement.

Hermetic; no LLM calls. Cost discipline is pre-registered (R-codes in pilot
memo), so these tests are the contract: if cap math or share math regresses,
the pilot could blow through budget undetected.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from mcparena.pilot import costs


@pytest.fixture(autouse=True)
def _reset_costs() -> None:
    costs.reset()


def test_initial_state_is_zero() -> None:
    state = costs.get_state()
    assert state.total_usd == 0.0
    assert state.program_usd == 0.0
    assert state.reflection_usd == 0.0
    assert state.reflection_share == 0.0


def test_add_program_cost_only() -> None:
    state = costs.get_state()
    # 1M input × $0.071 + 1M output × $0.10 = $0.171
    cost = state.add(1_000_000, 1_000_000, role="program", condition="baseline")
    assert cost == pytest.approx(0.171)
    assert state.program_usd == pytest.approx(0.171)
    assert state.reflection_usd == 0.0
    assert state.reflection_share == 0.0


def test_add_reflection_cost_attributes_correctly() -> None:
    state = costs.get_state()
    state.add(100_000, 100_000, role="program", condition="gepa")
    state.add(100_000, 100_000, role="reflection", condition="gepa")
    assert state.program_usd == pytest.approx(0.0171)
    assert state.reflection_usd == pytest.approx(0.0171)
    assert state.total_usd == pytest.approx(0.0342)
    assert state.reflection_share == pytest.approx(0.5)


def test_by_condition_breakdown() -> None:
    state = costs.get_state()
    state.add(1_000_000, 0, role="program", condition="baseline")
    state.add(1_000_000, 0, role="program", condition="miprov2")
    state.add(1_000_000, 0, role="program", condition="baseline")
    assert state.by_condition["baseline"] == pytest.approx(0.142)
    assert state.by_condition["miprov2"] == pytest.approx(0.071)


def test_check_caps_under_limit_passes() -> None:
    costs.get_state().add(100, 100, role="program", condition="baseline")
    costs.check_cost_caps()


def test_check_caps_raises_over_hard_cap() -> None:
    # 4B output × $0.10/M = $400 > $300 cap
    costs.get_state().add(0, 4_000_000_000, role="program", condition="baseline")
    with pytest.raises(RuntimeError, match="Cost cap exceeded"):
        costs.check_cost_caps()


def test_check_caps_raises_over_reflection_share() -> None:
    state = costs.get_state()
    state.add(1_000, 1_000, role="program", condition="gepa")
    state.add(9_000, 9_000, role="reflection", condition="gepa")
    with pytest.raises(RuntimeError, match="Reflection share"):
        costs.check_cost_caps()


def test_absorb_lm_history_drains_and_sums() -> None:
    """History must be drained on absorb — otherwise the next call double-counts."""
    fake_lm = SimpleNamespace(
        history=[
            {"usage": {"prompt_tokens": 1_000, "completion_tokens": 500}},
            {"usage": {"prompt_tokens": 2_000, "completion_tokens": 1_000}},
        ]
    )
    delta = costs.absorb_lm_history(fake_lm, role="program", condition="baseline")
    # 3000 × $0.071/M + 1500 × $0.10/M = 0.000213 + 0.00015 = 0.000363
    assert delta == pytest.approx(0.000363)
    assert fake_lm.history == []
    delta2 = costs.absorb_lm_history(fake_lm, role="program", condition="baseline")
    assert delta2 == 0.0


def test_absorb_lm_history_handles_missing_history() -> None:
    fake_lm: Any = SimpleNamespace()
    assert costs.absorb_lm_history(fake_lm, role="program", condition="baseline") == 0.0


def test_absorb_lm_history_handles_nested_usage_shapes() -> None:
    """DSPy history entries may nest `usage` under `response` / `raw`."""
    fake_lm = SimpleNamespace(
        history=[
            {"response": {"usage": {"prompt_tokens": 1_000, "completion_tokens": 1_000}}},
        ]
    )
    delta = costs.absorb_lm_history(fake_lm, role="program", condition="baseline")
    # 1000 × $0.071/M + 1000 × $0.10/M = 0.000071 + 0.0001 = 0.000171
    assert delta == pytest.approx(0.000171)


def test_reset_zeros_state_across_invocations() -> None:
    state = costs.get_state()
    state.add(100_000, 100_000, role="program", condition="baseline")
    assert costs.get_state().total_usd > 0
    costs.reset()
    assert costs.get_state().total_usd == 0.0
    assert costs.get_state().by_condition == {}
