"""ServerSpec pydantic validation."""

import pytest
from pydantic import ValidationError

from mcparena.pilot import ServerSpec


def test_minimal_valid_spec() -> None:
    spec = ServerSpec(
        name="test",
        mcp_bench_id="test-bench-id",
        baseline_score_mcp_bench=0.5,
        difficulty_tier="medium",
        stratification_rationale="test",
        transport="stdio",
        command="echo",
        args=["hello"],
    )
    assert spec.env == {}
    assert spec.notes == ""


def test_invalid_difficulty_tier_rejected() -> None:
    with pytest.raises(ValidationError):
        ServerSpec(
            name="x",
            mcp_bench_id="x",
            baseline_score_mcp_bench=0.0,
            difficulty_tier="impossible",  # type: ignore[arg-type]
            stratification_rationale="x",
            transport="stdio",
            command="x",
            args=[],
        )


def test_missing_required_field_rejected() -> None:
    with pytest.raises(ValidationError):
        ServerSpec(  # type: ignore[call-arg]
            name="x",
            # missing mcp_bench_id and everything else
        )


def test_json_round_trip() -> None:
    original = ServerSpec(
        name="round-trip",
        mcp_bench_id="abc",
        baseline_score_mcp_bench=0.42,
        difficulty_tier="hard",
        stratification_rationale="for the test",
        transport="stdio",
        command="cmd",
        args=["a", "b"],
        env={"KEY": "value"},
        notes="some note",
    )
    blob = original.model_dump_json()
    restored = ServerSpec.model_validate_json(blob)
    assert restored == original
