"""Unit tests for `mcparena.pilot.lm.get_lm`.

Verifies the OpenRouter vs Anthropic-direct routing branch, the
program/reflection role temperature + max_tokens fork, and registry
isolation (no cross-test leakage of cached `dspy.LM` instances).

Uses monkeypatch to spy on `dspy.LM` construction without making network calls.
"""

from __future__ import annotations

from typing import Any

import dspy
import pytest

from mcparena.pilot import lm as lm_module


@pytest.fixture(autouse=True)
def _clear_lm_registry() -> None:
    """Reset the module-level `_REGISTRY` between tests for isolation."""
    lm_module._REGISTRY.clear()


@pytest.fixture
def spy_dspy_lm(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Replace `dspy.LM(...)` with a spy that records constructor kwargs."""
    calls: list[dict[str, Any]] = []

    class _StubLM:
        def __init__(self, **kwargs: Any) -> None:
            calls.append(kwargs)

    monkeypatch.setattr(dspy, "LM", _StubLM)
    return calls


def test_default_uses_openrouter_dated_slug(
    spy_dspy_lm: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    lm_module.get_lm()
    assert len(spy_dspy_lm) == 1
    call = spy_dspy_lm[0]
    assert call["model"] == "openrouter/qwen/qwen3-235b-a22b-2507"
    assert call["api_key"] == "sk-or-test"
    assert call["api_base"] == "https://openrouter.ai/api/v1"
    assert call["cache"] is False  # P0 fix — trials must NOT be cached


def test_openrouter_prefix_reads_openrouter_key(
    spy_dspy_lm: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-1")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-1")
    lm_module.get_lm("openrouter/anthropic/claude-sonnet-4")
    assert spy_dspy_lm[0]["api_key"] == "sk-or-1"


def test_anthropic_direct_reads_anthropic_key_and_omits_api_base(
    spy_dspy_lm: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-1")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-1")
    lm_module.get_lm("anthropic/claude-sonnet-4-6")
    call = spy_dspy_lm[0]
    assert call["api_key"] == "sk-ant-1"
    assert "api_base" not in call


def test_program_role_sets_low_temperature_and_smaller_window(
    spy_dspy_lm: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "x")
    lm_module.get_lm(role="program")
    call = spy_dspy_lm[0]
    assert call["temperature"] == 0.7
    assert call["max_tokens"] == 8192


def test_reflection_role_sets_high_temperature_and_larger_window(
    spy_dspy_lm: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "x")
    lm_module.get_lm(role="reflection")
    call = spy_dspy_lm[0]
    assert call["temperature"] == 1.0
    assert call["max_tokens"] == 16000


def test_invalid_role_raises_value_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "x")
    with pytest.raises(ValueError, match="role must be"):
        lm_module.get_lm(role="refelction")  # type: ignore[arg-type]


def test_registry_caches_per_model_role(
    spy_dspy_lm: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "x")
    lm_module.get_lm(role="program")
    lm_module.get_lm(role="program")  # cache hit
    lm_module.get_lm(role="reflection")  # cache miss (different role)
    assert len(spy_dspy_lm) == 2  # not 3 — first program call cached
