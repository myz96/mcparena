"""Judge tests: syrupy snapshot of Assess Signature + descriptive calibration.

The Assess Signature snapshot guards against unintentional drift in the judge
prompt (its docstring + field descriptions ARE the prompt the judge LM sees).
Snapshot updates require deliberate intent — fail loudly when the judge prompt
changes silently.

The calibration test is `@pytest.mark.live` (opt-in via `pytest -m live`) since
it makes real API calls. It reports agreement on the 10-transcript fixture
descriptively; per plan v5.1 it is NOT a hard runtime gate.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from mcparena.pilot import Assess

FIXTURE = Path(__file__).parent / "judge_fixtures" / "transcripts.jsonl"


def _format_assess_signature() -> str:
    """Stable string representation of the Assess Signature for snapshot diffing."""
    lines: list[str] = []
    lines.append(f"class: {Assess.__name__}")
    lines.append(f"docstring: {Assess.__doc__}")
    lines.append("fields:")
    for field_name in sorted(Assess.model_fields.keys()):
        field = Assess.model_fields[field_name]
        # json_schema_extra holds dspy's input/output marker + desc
        extra = field.json_schema_extra if isinstance(field.json_schema_extra, dict) else {}
        kind = extra.get("__dspy_field_type", "?")
        desc = extra.get("desc", field.description or "")
        lines.append(f"  - {field_name} [{kind}] {desc}")
    return "\n".join(lines)


def test_assess_signature_snapshot(snapshot: Any) -> None:
    """Snapshot the Assess Signature. CI fails on any drift — update with intent."""
    assert _format_assess_signature() == snapshot


def test_judge_fixture_well_formed() -> None:
    """The 10-transcript calibration fixture parses cleanly + has expected balance."""
    records = [json.loads(line) for line in FIXTURE.read_text().splitlines() if line.strip()]
    assert len(records) == 10, "v5.1 pilot ships 10 hand-labeled transcripts"
    pass_count = sum(1 for r in records if r["expected"] is True)
    fail_count = sum(1 for r in records if r["expected"] is False)
    assert pass_count == 5 and fail_count == 5, (
        f"Expected balanced 5/5 pass-fail; got {pass_count}/{fail_count}"
    )
    required = {"id", "user_request", "trajectory", "final_answer", "expected_outcome", "expected"}
    for r in records:
        assert required <= set(r.keys()), f"missing fields in {r.get('id', '?')}"


@pytest.mark.live
def test_judge_calibration_descriptive() -> None:
    """Live judge agreement on the 10-transcript fixture (descriptive, not gating).

    Requires ANTHROPIC_API_KEY. Reports agreement N/10. Pilot memo includes this
    number for context but does NOT block on it (per plan v5.1 R4.5 demotion).
    """
    import dspy

    from mcparena.pilot import Assess, get_lm

    dspy.configure(lm=get_lm("anthropic/claude-sonnet-4-6"))
    judge = dspy.Predict(Assess)

    records = [json.loads(line) for line in FIXTURE.read_text().splitlines() if line.strip()]
    agreements = 0
    for r in records:
        result = judge(
            user_request=r["user_request"],
            trajectory=r["trajectory"],
            final_answer=r["final_answer"],
            expected_outcome=r["expected_outcome"],
        )
        if bool(result.success) == bool(r["expected"]):
            agreements += 1

    # Descriptive only — log result, no assertion (per v5.1 R4.5)
    print(f"\nJudge agreement on 10-transcript fixture: {agreements}/10")
