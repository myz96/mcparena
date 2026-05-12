"""Language model registry for the mcparena pilot.

Pilot uses Claude Sonnet 4 (the model MCP-Bench tested in their published
leaderboard) routed through OpenRouter, so a single ``OPENROUTER_API_KEY``
covers Anthropic, OpenAI, Google, and any other provider Phase 1 adds.

Both pilot roles use the same underlying model id; only temperature and
max_tokens differ:

- Program / judge LM : temperature 0.7, max_tokens 8192
- Reflection LM (GEPA): temperature 1.0, max_tokens 16000

Default model id: dated OpenRouter slug pinned to the version MCP-Bench
tested. Phase 1 wires multi-model and cross-model judging through this same
registry.
"""

from __future__ import annotations

import os
from typing import Any, Literal

# OpenRouter exposes Anthropic models via semver aliases (anthropic/claude-sonnet-4,
# /4.5, /4.6) — not Anthropic's internal dated slugs. We pin to "claude-sonnet-4"
# because that's exactly what MCP-Bench tested (score 0.681 on their leaderboard).
# Verified live via openrouter.ai/api/v1/models on 2026-05-13.
DEFAULT_PILOT_MODEL = "openrouter/anthropic/claude-sonnet-4"
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"

Role = Literal["program", "reflection"]

# Distinct registry keys per (model_id, role) so reflection and program get
# separately configured `dspy.LM` instances even when the underlying model
# is the same.
_REGISTRY: dict[tuple[str, Role], Any] = {}


def get_lm(model_id: str = DEFAULT_PILOT_MODEL, role: Role = "program") -> Any:
    """Return a configured `dspy.LM` for the given (model_id, role).

    role:
      - "program"    : program / judge LM — temperature 0.7, max_tokens 8192
      - "reflection" : GEPA reflection LM  — temperature 1.0, max_tokens 16000

    Routes through OpenRouter by default (``OPENROUTER_API_KEY``). Pass an
    Anthropic-direct id like ``"anthropic/claude-sonnet-4-6"`` to bypass
    OpenRouter and use ``ANTHROPIC_API_KEY`` instead.

    Raises ``ValueError`` if ``role`` is not in {"program", "reflection"}
    (defensive: catches typos like "refelction" that would silently route to
    the wrong temperature / max_tokens).
    """
    if role not in ("program", "reflection"):
        raise ValueError(f"role must be 'program' or 'reflection', got {role!r}")

    key = (model_id, role)
    if key not in _REGISTRY:
        import dspy

        is_openrouter = model_id.startswith("openrouter/")
        api_key_env = "OPENROUTER_API_KEY" if is_openrouter else "ANTHROPIC_API_KEY"
        is_reflection = role == "reflection"

        kwargs: dict[str, Any] = {
            "model": model_id,
            "api_key": os.environ.get(api_key_env),
            "temperature": 1.0 if is_reflection else 0.7,
            "max_tokens": 16000 if is_reflection else 8192,
        }
        if is_openrouter:
            # LiteLLM sometimes fails to auto-resolve the OpenRouter base for
            # the `openrouter/` provider prefix depending on version. Explicit
            # api_base is cheap insurance against routing surprises.
            kwargs["api_base"] = OPENROUTER_API_BASE

        _REGISTRY[key] = dspy.LM(**kwargs)
    return _REGISTRY[key]
