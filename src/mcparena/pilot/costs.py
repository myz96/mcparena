"""Cost tracking for the mcparena pilot."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

# Pricing for `anthropic/claude-sonnet-4` via OpenRouter, effective 2026-05-13.
SONNET_4_INPUT_USD_PER_M = 3.0
SONNET_4_OUTPUT_USD_PER_M = 15.0
HARD_CAP_USD = 300.0
REFLECTION_SHARE_GATE = 0.60

Role = Literal["program", "reflection"]


@dataclass
class CostState:
    total_usd: float = 0.0
    program_usd: float = 0.0
    reflection_usd: float = 0.0
    by_condition: dict[str, float] = field(default_factory=dict)

    def add(self, prompt_tokens: int, completion_tokens: int, role: Role, condition: str) -> float:
        cost = (
            prompt_tokens * SONNET_4_INPUT_USD_PER_M / 1_000_000
            + completion_tokens * SONNET_4_OUTPUT_USD_PER_M / 1_000_000
        )
        self.total_usd += cost
        if role == "reflection":
            self.reflection_usd += cost
        else:
            self.program_usd += cost
        self.by_condition[condition] = self.by_condition.get(condition, 0.0) + cost
        return cost

    @property
    def reflection_share(self) -> float:
        return self.reflection_usd / self.total_usd if self.total_usd > 0 else 0.0


_state = CostState()


def get_state() -> CostState:
    return _state


def reset() -> None:
    global _state
    _state = CostState()


def check_cost_caps() -> None:
    """Raise on cumulative >$300 OR reflection share >60% (pre-reg cost discipline)."""
    if _state.total_usd > HARD_CAP_USD:
        raise RuntimeError(
            f"Cost cap exceeded: ${_state.total_usd:.2f} > ${HARD_CAP_USD:.2f}. "
            "Halt the runner; investigate before resuming."
        )
    if _state.reflection_usd > 0 and _state.reflection_share > REFLECTION_SHARE_GATE:
        raise RuntimeError(
            f"Reflection share {_state.reflection_share:.1%} exceeds "
            f"{REFLECTION_SHARE_GATE:.0%}. Drop GEPA on remaining servers."
        )


def absorb_lm_history(lm: Any, role: Role, condition: str) -> float:
    """Sum tokens from `lm.history` into the global state and clear history.

    Call AFTER each condition completes — clearing history prevents the next
    absorb() from double-counting.
    """
    history = getattr(lm, "history", None)
    if history is None:
        return 0.0

    delta = 0.0
    for entry in history:
        usage = _extract_usage(entry)
        prompt = int(usage.get("prompt_tokens", 0) or 0)
        completion = int(usage.get("completion_tokens", 0) or 0)
        if prompt or completion:
            delta += _state.add(prompt, completion, role, condition)
    history.clear()
    return delta


def _extract_usage(entry: Any) -> dict[str, Any]:
    """Locate token-usage info across plausible DSPy LM history shapes."""
    if not isinstance(entry, dict):
        return {}
    if "usage" in entry and isinstance(entry["usage"], dict):
        return entry["usage"]
    # Some DSPy versions nest under "response" / "raw" / "completion".
    for key in ("response", "raw", "completion"):
        nested = entry.get(key)
        if isinstance(nested, dict) and "usage" in nested:
            usage = nested["usage"]
            if isinstance(usage, dict):
                return usage
    return {}
