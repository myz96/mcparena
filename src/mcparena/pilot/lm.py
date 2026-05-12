"""Language model registry for the mcparena pilot.

Pilot LMs:
- `anthropic/claude-sonnet-4-6` — program LM (the ReAct agent) AND judge LM.
- `anthropic/claude-opus-4-7`   — reflection LM for GEPA only. Higher max_tokens
                                  and temperature=1.0 per GEPA's docs.

Phase 1 wires multi-model and cross-model judging through the same registry.
"""

from __future__ import annotations

import os
from typing import Any

_REGISTRY: dict[str, Any] = {}


def get_lm(model_id: str) -> Any:
    """Return a configured `dspy.LM` for the given model id (creates on first call)."""
    if model_id not in _REGISTRY:
        import dspy  # lazy: dspy is heavy; deferred until first pilot LM access

        is_reflection = "opus" in model_id  # Opus is reflection LM in pilot
        _REGISTRY[model_id] = dspy.LM(
            model_id,
            api_key=os.environ.get("ANTHROPIC_API_KEY"),
            temperature=1.0 if is_reflection else 0.7,
            max_tokens=16000 if is_reflection else 8192,
        )
    return _REGISTRY[model_id]
