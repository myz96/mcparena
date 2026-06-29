"""Pure functions for human-alignment calibration of the verifier.

Guards against proxy / Goodhart misalignment: does the programmatic verifier
agree with a human's notion of task success? These are I/O-free and unit-tested;
the interactive blind-grading workflow lives in scripts/align_calibrate.py.

A trial's verifier verdict is binarized as PASS iff score == 1.0 (full task
success), to compare apples-to-apples with a human's binary pass/fail.
"""

from __future__ import annotations

import random
from typing import Any


def verifier_pass(score: float) -> int:
    """Binarize the verifier's (possibly fractional) score to task success."""
    return 1 if score >= 1.0 else 0


def agreement(pairs: list[tuple[int, int]]) -> float:
    """Raw agreement rate between two binary raters."""
    if not pairs:
        return 0.0
    return sum(1 for a, b in pairs if a == b) / len(pairs)


def cohen_kappa(pairs: list[tuple[int, int]]) -> float:
    """Cohen's κ for two binary raters — corrects raw agreement for chance.

    κ = (po - pe) / (1 - pe). Degenerate case (both raters constant and equal)
    returns 1.0; constant-but-disagreeing returns 0.0.
    """
    n = len(pairs)
    if n == 0:
        return 0.0
    po = agreement(pairs)
    pa1 = sum(a for a, _ in pairs) / n
    pb1 = sum(b for _, b in pairs) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    if pe >= 1.0:  # both raters constant and identical
        return 1.0 if po >= 1.0 else 0.0
    return (po - pe) / (1 - pe)


def confusion(pairs: list[tuple[int, int]]) -> dict[str, int]:
    """Confusion counts keyed verifier×human (e.g. 'verifierPASS_humanFAIL')."""
    out = {
        "verifierPASS_humanPASS": 0,
        "verifierPASS_humanFAIL": 0,
        "verifierFAIL_humanPASS": 0,
        "verifierFAIL_humanFAIL": 0,
    }
    for v, h in pairs:
        vk = "PASS" if v else "FAIL"
        hk = "PASS" if h else "FAIL"
        out[f"verifier{vk}_human{hk}"] += 1
    return out


def stratified_sample(trials: list[dict[str, Any]], n: int, seed: int = 0) -> list[dict[str, Any]]:
    """Pick n trials balanced between verifier-pass / verifier-fail, spread across kinds.

    Don't waste a small human budget on easy agreements: aim n/2 fails / n/2
    passes, round-robin across task kinds within each bucket. Falls back to the
    other bucket if one is too small.
    """
    rng = random.Random(seed)
    passed = [t for t in trials if verifier_pass(t.get("score", 0.0))]
    failed = [t for t in trials if not verifier_pass(t.get("score", 0.0))]

    def _spread(bucket: list[dict[str, Any]], want: int) -> list[dict[str, Any]]:
        by_kind: dict[str, list[dict[str, Any]]] = {}
        for t in bucket:
            by_kind.setdefault(t.get("kind", "?"), []).append(t)
        for v in by_kind.values():
            rng.shuffle(v)
        picked: list[dict[str, Any]] = []
        kinds = sorted(by_kind)
        while len(picked) < want and any(by_kind[k] for k in kinds):
            for k in kinds:
                if by_kind[k] and len(picked) < want:
                    picked.append(by_kind[k].pop())
        return picked

    want_fail = min(len(failed), n // 2)
    want_pass = min(len(passed), n - want_fail)
    want_fail = min(len(failed), n - want_pass)  # backfill if passes were short
    sample = _spread(failed, want_fail) + _spread(passed, want_pass)
    rng.shuffle(sample)
    return sample[:n]


def gate(
    pairs: list[tuple[int, int]], kappa_min: float = 0.6, agree_min: float = 0.8
) -> dict[str, Any]:
    """Calibration gate: κ ≥ kappa_min AND agreement ≥ agree_min."""
    k = cohen_kappa(pairs)
    a = agreement(pairs)
    return {
        "n": len(pairs),
        "agreement": round(a, 3),
        "cohen_kappa": round(k, 3),
        "confusion": confusion(pairs),
        "passed_gate": bool(k >= kappa_min and a >= agree_min),
        "thresholds": {"kappa_min": kappa_min, "agree_min": agree_min},
    }
