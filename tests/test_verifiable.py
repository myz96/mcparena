"""Hermetic tests for the verifiable-task verifier (no server / no LLM).

Construct task dicts with known ground truth and assert verify() scores them
correctly — including format-robustness (so we don't re-introduce the
format-gating that wrecked the original pilot).
"""

from __future__ import annotations

import json

from mcparena.verifiable.tasks import verify


def _t(kind: str, gt: dict, tol: float = 1e-3) -> dict:
    return {"task_id": f"t_{kind}", "kind": kind, "ground_truth": gt, "tolerance_rel": tol}


def test_single_correct_and_wrong() -> None:
    t = _t("single", {"converted_value": 176.6667})
    assert verify(t, "The result is 176.6667 celsius.")[0] == 1.0
    assert verify(t, "The result is 0.123.")[0] == 0.0
    assert verify(t, "")[0] == 0.0


def test_chain_prefers_final_number() -> None:
    t = _t("chain", {"final_value": 613.47})
    # Answer shows intermediate work; the LAST number is the reported final.
    assert verify(t, "step1=1640.42 ft; step2=1840.42 ft; final = 613.47 yards")[0] == 1.0
    assert verify(t, "final = 999.9")[0] == 0.0


def test_aggregate_scalar() -> None:
    t = _t("aggregate", {"aggregate": 1100812418720416.9, "op": "sum"})
    assert verify(t, "SUM = 1100812418720416.9 bits")[0] == 1.0
    assert verify(t, "SUM = 1.0e15")[0] == 0.0


def test_conditional_scalar() -> None:
    t = _t("conditional", {"final_value": 143924354088.96, "branch_unit": "megabytes"})
    assert verify(t, "Result: 143924354088.96 megabytes")[0] == 1.0
    assert verify(t, "Result: 42")[0] == 0.0


def _multi_gt() -> dict:
    return {
        "items": [
            {"name": "sensor_1", "converted_value": 100.0, "status": "PASS"},
            {"name": "sensor_2", "converted_value": 2.5, "status": "FAIL"},
        ]
    }


def test_multi_all_correct() -> None:
    t = _t("multi_hard", _multi_gt())
    ans = json.dumps(
        [
            {"name": "sensor_1", "converted_value": 100.0, "status": "PASS"},
            {"name": "sensor_2", "converted_value": 2.5, "status": "FAIL"},
        ]
    )
    assert verify(t, ans)[0] == 1.0


def test_multi_partial_and_wrong_status() -> None:
    t = _t("multi_hard", _multi_gt())
    # sensor_2 status wrong → only 1/2 correct.
    ans = json.dumps(
        [
            {"name": "sensor_1", "converted_value": 100.0, "status": "PASS"},
            {"name": "sensor_2", "converted_value": 2.5, "status": "PASS"},
        ]
    )
    assert verify(t, ans)[0] == 0.5


def test_multi_format_robust_wrapped_array() -> None:
    # Content correct but wrapped in an object — verifier must still find the array
    # (we check VALUES, not the JSON wrapper — no format-gating).
    t = _t("multi_hard", _multi_gt())
    ans = '{"results": [{"name": "sensor_1", "converted_value": 100.0, "status": "PASS"}, {"name": "sensor_2", "converted_value": 2.5, "status": "FAIL"}]}'
    assert verify(t, ans)[0] == 1.0


def test_multi_missing_sensor() -> None:
    t = _t("multi_hard", _multi_gt())
    ans = json.dumps([{"name": "sensor_1", "converted_value": 100.0, "status": "PASS"}])
    score, fb = verify(t, ans)
    assert score == 0.5
    assert "missing" in fb


def test_tolerance_is_relative() -> None:
    t = _t("single", {"converted_value": 1_000_000.0}, tol=1e-3)
    assert verify(t, "999500")[0] == 1.0  # within 0.1%
    assert verify(t, "990000")[0] == 0.0  # 1% off
