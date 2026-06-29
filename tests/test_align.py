"""Hermetic tests for the alignment-calibration math (Cohen's κ, sampling, gate)."""

from __future__ import annotations

from mcparena.verifiable import align


def test_kappa_perfect_agreement() -> None:
    pairs = [(1, 1), (0, 0), (1, 1), (0, 0)]
    assert align.cohen_kappa(pairs) == 1.0
    assert align.agreement(pairs) == 1.0


def test_kappa_chance_level_is_zero() -> None:
    # 50/50 raters agreeing half the time → κ ≈ 0.
    pairs = [(1, 1), (0, 0), (1, 0), (0, 1)]
    assert abs(align.cohen_kappa(pairs)) < 1e-9


def test_kappa_constant_raters() -> None:
    assert align.cohen_kappa([(1, 1), (1, 1)]) == 1.0  # both always pass, agree
    assert align.cohen_kappa([(1, 0), (1, 0)]) == 0.0  # constant but disagree


def test_confusion_counts() -> None:
    pairs = [(1, 1), (1, 0), (0, 1), (0, 0)]
    c = align.confusion(pairs)
    assert c["verifierPASS_humanPASS"] == 1
    assert c["verifierPASS_humanFAIL"] == 1
    assert c["verifierFAIL_humanPASS"] == 1
    assert c["verifierFAIL_humanFAIL"] == 1


def test_gate_thresholds() -> None:
    good = [(1, 1)] * 9 + [(0, 0)]  # near-perfect
    assert align.gate(good)["passed_gate"] is True
    bad = [(1, 0)] * 5 + [(0, 1)] * 5  # systematic disagreement
    assert align.gate(bad)["passed_gate"] is False


def test_stratified_sample_balances_pass_fail() -> None:
    trials = [{"score": 1.0, "kind": "chain", "task_id": f"p{i}"} for i in range(8)] + [
        {"score": 0.0, "kind": "multi_hard", "task_id": f"f{i}"} for i in range(8)
    ]
    sample = align.stratified_sample(trials, n=6, seed=1)
    assert len(sample) == 6
    n_pass = sum(1 for t in sample if align.verifier_pass(t["score"]))
    assert n_pass == 3  # balanced 3 pass / 3 fail


def test_verifier_pass_binarization() -> None:
    assert align.verifier_pass(1.0) == 1
    assert align.verifier_pass(0.93) == 0  # partial credit = task not fully successful
    assert align.verifier_pass(0.0) == 0
