"""mcparena pilot — validates DSPy improves MCP server tool-use.

Public surface:
- PILOT_SERVERS   : list of stratified ServerSpec targets (populated Unit 5)
- TASKS_BY_SERVER : dict mapping server name -> list of dspy.Example tasks
- Assess          : dspy.Signature for the cross-judge prompt
- judge_metric_evaluate / judge_metric_gepa : metric functions for the two
                    DSPy optimizer surfaces (3-arg and 5-arg)
- get_lm          : LM registry (program/judge = Sonnet 4.6, reflection = Opus 4.7)

Behavior wiring lives in `mcparena.pilot.pilot.main` and is invoked from the
CLI entry point (`mcparena pilot ...`).
"""

from mcparena.pilot.judge import Assess, judge_metric_evaluate, judge_metric_gepa
from mcparena.pilot.lm import get_lm
from mcparena.pilot.tasks import PILOT_SERVERS, TASKS_BY_SERVER, Condition, ServerSpec

__all__ = [
    "PILOT_SERVERS",
    "TASKS_BY_SERVER",
    "Assess",
    "Condition",
    "ServerSpec",
    "get_lm",
    "judge_metric_evaluate",
    "judge_metric_gepa",
]
