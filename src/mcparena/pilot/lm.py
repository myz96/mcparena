"""Language model registry for the mcparena pilot.

Pilot uses Claude Sonnet 4.6 for everything (v5.1 update: dropped Opus 4.7
for GEPA reflection to halve cost). Both roles use the same model id; only
temperature and max_tokens differ:

- Program / judge LM : temperature 0.7, max_tokens 8192
- Reflection LM (GEPA): temperature 1.0, max_tokens 16000

Phase 1 wires multi-model and cross-model judging through this same registry.
"""

from __future__ import annotations

import os
from typing import Any

# Distinct registry keys per (model_id, role) so reflection and program get
# separately configured `dspy.LM` instances even when the underlying model
# is the same.
_REGISTRY: dict[tuple[str, str], Any] = {}


def get_lm(model_id: str, role: str = "program") -> Any:
    """Return a configured `dspy.LM` for the given (model_id, role).

    role:
      - "program"    : program / judge LM (Sonnet 4.6 in pilot) — temperature 0.7, max_tokens 8192
      - "reflection" : GEPA reflection LM (Sonnet 4.6 in pilot)  — temperature 1.0, max_tokens 16000
    """
    key = (model_id, role)
    if key not in _REGISTRY:
        import dspy  # lazy: dspy is heavy; deferred until first pilot LM access

        is_reflection = role == "reflection"
        _REGISTRY[key] = dspy.LM(
            model_id,
            api_key=os.environ.get("ANTHROPIC_API_KEY"),
            temperature=1.0 if is_reflection else 0.7,
            max_tokens=16000 if is_reflection else 8192,
        )
    return _REGISTRY[key]
