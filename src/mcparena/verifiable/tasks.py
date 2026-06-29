"""Verifiable task generation + programmatic verification for the unit-converter MCP.

Backwards-from-execution ("the server is the oracle"): sample params → EXECUTE
against the server → capture the real result as ground truth → template a task →
verify the agent's answer programmatically (tolerance match). No LLM judge.

Task kinds, easy → hard:
  - single      : one conversion (saturation control)
  - multi       : K independent conversions + PASS/FAIL vs threshold (batchable)
  - chain       : convert → arithmetic → convert (sequential dependency; not batchable)
  - aggregate   : convert K readings to a common unit, return SUM/MAX
  - conditional : convert X; branch on the result; convert Y via the chosen unit
  - multi_hard  : 12–18 mixed-type items, some with AMBIGUOUS units (gallon vs
                  gallon (US) vs gallon (imperial)) where the wrong pick is plausible
"""

from __future__ import annotations

import json
import random
import re
from collections.abc import Callable
from typing import Any

TOLERANCE_REL = 1e-3

# Types whose values are safe to sample positively in [1, 1000].
SAFE_TYPES = [
    "length",
    "mass",
    "volume",
    "area",
    "energy",
    "force",
    "pressure",
    "power",
    "speed",
    "computer_data",
    "density",
    "time",
    "angle",
]

# Default hard mix (no singles/easy-multi — they saturate).
DEFAULT_HARD_COUNTS = {"chain": 4, "aggregate": 3, "conditional": 3, "multi_hard": 4}


def uc_stdio_params(repo_root: Any) -> Any:
    """StdioServerParameters for the unit-converter MCP (repo_root: pathlib.Path)."""
    from mcp.client.stdio import StdioServerParameters

    return StdioServerParameters(
        command="uv",
        args=["run", "unit-converter-mcp"],
        env={},
        cwd=str(repo_root / "third_party/mcp-bench-tasks/mcp_servers/unit-converter-mcp"),
    )


# ----------------------------- server oracle -----------------------------


def _supported_units(session: Any) -> dict[str, list[str]]:
    data: Any = json.loads(session.call_tool("list_supported_units"))
    return {str(k): [str(u) for u in v] for k, v in data.items()}


def _convert(session: Any, ctype: str, value: float, from_unit: str, to_unit: str) -> float:
    """Execute one conversion against the server; return the converted value (ground truth)."""
    raw = session.call_tool(f"convert_{ctype}", value=value, from_unit=from_unit, to_unit=to_unit)
    return float(json.loads(raw)["converted_value"])


def _ambiguous_units(units: list[str]) -> list[str]:
    """Units that belong to a family with >1 variant (e.g. gallon / gallon (US) / gallon (imperial))."""
    base: dict[str, list[str]] = {}
    for u in units:
        key = re.sub(r"\s*\(.*\)", "", u).strip()
        base.setdefault(key, []).append(u)
    return [u for fam in base.values() if len(fam) > 1 for u in fam]


def _pick_unit(rng: random.Random, units: list[str], ambiguous: bool) -> str:
    if ambiguous:
        amb = _ambiguous_units(units)
        if amb:
            return rng.choice(amb)
    return rng.choice(units)


# ----------------------------- generators -----------------------------


def _types(supported: dict[str, list[str]], min_units: int = 2) -> list[str]:
    return [t for t in SAFE_TYPES if t in supported and len(supported[t]) >= min_units]


def _gen_single(
    session: Any, rng: random.Random, supported: dict[str, list[str]], i: int
) -> dict[str, Any]:
    ctype = rng.choice(_types(supported))
    a, b = rng.sample(supported[ctype], 2)
    v = round(rng.uniform(1, 1000), 2)
    gt = _convert(session, ctype, v, a, b)
    return {
        "task_id": f"uc_single_{i:03d}",
        "kind": "single",
        "difficulty": "easy",
        "user_request": f"Convert {v} {a} to {b} using the unit-converter tools. Respond with the numeric converted value.",
        "ground_truth": {"converted_value": gt},
        "tolerance_rel": TOLERANCE_REL,
    }


def _gen_multi(
    session: Any,
    rng: random.Random,
    supported: dict[str, list[str]],
    i: int,
    k_range: tuple[int, int] = (4, 7),
    ambiguous: bool = False,
    hard: bool = False,
) -> dict[str, Any]:
    k = rng.randint(*k_range)
    items = []
    for j in range(k):
        ctype = rng.choice(_types(supported))
        a = _pick_unit(rng, supported[ctype], ambiguous)
        b = _pick_unit(rng, [u for u in supported[ctype] if u != a], ambiguous)
        v = round(rng.uniform(1, 1000), 2)
        conv = _convert(session, ctype, v, a, b)
        # Threshold clearly above or below the converted value (avoid the
        # [0.85, 1.15] knife-edge band where PASS/FAIL is ambiguous ground truth).
        factor = rng.uniform(0.5, 0.85) if rng.random() < 0.5 else rng.uniform(1.15, 1.5)
        thr = round(conv * factor, 4)
        items.append(
            {
                "name": f"sensor_{j + 1}",
                "value": v,
                "from_unit": a,
                "to_unit": b,
                "threshold": thr,
                "gt_conv": conv,
                "gt_status": "PASS" if conv >= thr else "FAIL",
            }
        )
    lines = "\n".join(
        f"  - {it['name']}: {it['value']} {it['from_unit']} -> {it['to_unit']} (threshold {it['threshold']} {it['to_unit']})"
        for it in items
    )
    return {
        "task_id": f"uc_{'multihard' if hard else 'multi'}_{i:03d}",
        "kind": "multi_hard" if hard else "multi",
        "difficulty": "hard" if hard else "medium",
        "user_request": (
            f"You are validating {k} sensor readings. For EACH reading: convert the value to its target "
            f"unit using the unit-converter tools, then mark PASS if the converted value >= its threshold, "
            f"else FAIL. Use the EXACT unit named (e.g. 'gallon (imperial)' is not 'gallon').\n\n"
            f"Readings:\n{lines}\n\nReturn a JSON array; one object per sensor with keys: name, converted_value, status."
        ),
        "ground_truth": {
            "items": [
                {"name": it["name"], "converted_value": it["gt_conv"], "status": it["gt_status"]}
                for it in items
            ]
        },
        "tolerance_rel": TOLERANCE_REL,
    }


def _gen_chain(
    session: Any, rng: random.Random, supported: dict[str, list[str]], i: int
) -> dict[str, Any]:
    ctype = rng.choice(_types(supported, min_units=3))
    a, b, c = rng.sample(supported[ctype], 3)
    v = round(rng.uniform(1, 1000), 2)
    r1 = _convert(session, ctype, v, a, b)
    if rng.random() < 0.5:
        delta = round(rng.uniform(1, 500), 2)
        r2, step2 = r1 + delta, f"add {delta} {b} to that result"
    else:
        factor = rng.randint(2, 5)
        r2, step2 = r1 * factor, f"multiply that result by {factor}"
    final = _convert(session, ctype, r2, b, c)
    return {
        "task_id": f"uc_chain_{i:03d}",
        "kind": "chain",
        "difficulty": "hard",
        "user_request": (
            f"A sensor reads {v} {a}. Do these steps IN ORDER, each using the previous step's output:\n"
            f"  1. Convert {v} {a} to {b}.\n  2. {step2[0].upper()}{step2[1:]}.\n"
            f"  3. Convert the step-2 total from {b} to {c}.\nReport the final value in {c} (numeric)."
        ),
        "ground_truth": {"final_value": final},
        "tolerance_rel": TOLERANCE_REL,
    }


def _gen_aggregate(
    session: Any, rng: random.Random, supported: dict[str, list[str]], i: int
) -> dict[str, Any]:
    ctype = rng.choice(_types(supported))
    common = rng.choice(supported[ctype])
    k = rng.randint(4, 6)
    items, converted = [], []
    for _ in range(k):
        a = rng.choice([u for u in supported[ctype] if u != common])
        v = round(rng.uniform(1, 1000), 2)
        converted.append(_convert(session, ctype, v, a, common))
        items.append(f"{v} {a}")
    op = rng.choice(["sum", "max"])
    agg = sum(converted) if op == "sum" else max(converted)
    return {
        "task_id": f"uc_agg_{i:03d}",
        "kind": "aggregate",
        "difficulty": "hard",
        "user_request": (
            f"Convert each of these readings to {common} using the unit-converter tools, then report the "
            f"{op.upper()} of the converted values (numeric):\n  " + "; ".join(items)
        ),
        "ground_truth": {"aggregate": agg, "op": op},
        "tolerance_rel": TOLERANCE_REL,
    }


def _gen_conditional(
    session: Any, rng: random.Random, supported: dict[str, list[str]], i: int
) -> dict[str, Any]:
    ctype = rng.choice(_types(supported, min_units=3))
    a, b = rng.sample(supported[ctype], 2)
    d1, d2 = rng.sample(supported[ctype], 2)
    x = round(rng.uniform(1, 1000), 2)
    y = round(rng.uniform(1, 1000), 2)
    c = rng.choice(supported[ctype])
    r1 = _convert(session, ctype, x, a, b)
    thr = round(r1 * rng.uniform(0.5, 1.5), 4)
    chosen = d1 if r1 >= thr else d2
    final = _convert(session, ctype, y, c, chosen)
    return {
        "task_id": f"uc_cond_{i:03d}",
        "kind": "conditional",
        "difficulty": "hard",
        "user_request": (
            f"Step 1: convert {x} {a} to {b}. Step 2: if that result is >= {thr}, convert {y} {c} to {d1}; "
            f"otherwise convert {y} {c} to {d2}. Report the Step-2 converted value (numeric)."
        ),
        "ground_truth": {"final_value": final, "branch_unit": chosen},
        "tolerance_rel": TOLERANCE_REL,
    }


_GeneratorFn = Callable[[Any, random.Random, dict[str, list[str]], int], dict[str, Any]]

_GENERATORS: dict[str, _GeneratorFn] = {
    "single": _gen_single,
    "multi": _gen_multi,
    "chain": _gen_chain,
    "aggregate": _gen_aggregate,
    "conditional": _gen_conditional,
    "multi_hard": lambda s, r, u, i: _gen_multi(
        s, r, u, i, k_range=(16, 22), ambiguous=True, hard=True
    ),
}


def generate_tasks(
    session: Any, seed: int = 0, counts: dict[str, int] | None = None
) -> list[dict[str, Any]]:
    """Generate verifiable tasks by executing sampled conversions against the server."""
    rng = random.Random(seed)
    supported = _supported_units(session)
    counts = counts or dict(DEFAULT_HARD_COUNTS)
    tasks: list[dict[str, Any]] = []
    for kind, n in counts.items():
        gen = _GENERATORS[kind]
        for i in range(n):
            tasks.append(gen(session, rng, supported, i))
    return tasks


# ----------------------------- verifier -----------------------------


def _nums(text: str) -> list[float]:
    return [float(x) for x in re.findall(r"-?\d+\.?\d*(?:[eE][+-]?\d+)?", text or "")]


def _close(a: float, b: float, rel: float) -> bool:
    return abs(a - b) <= max(rel * abs(b), rel)


def _scalar_match(fa: str, target: float, rel: float) -> bool:
    nums = _nums(fa)
    if not nums:
        return False
    # Prefer the LAST number (the "report the final value" answer), else any.
    return _close(nums[-1], target, rel) or any(_close(n, target, rel) for n in nums)


def verify(task: dict[str, Any], final_answer: str) -> tuple[float, str]:
    """Programmatic check → (score in [0,1], feedback string for GEPA)."""
    rel = task.get("tolerance_rel", TOLERANCE_REL)
    fa = final_answer or ""
    kind = task["kind"]

    if kind == "single":
        t = task["ground_truth"]["converted_value"]
        return (
            (1.0, f"Correct ({t:.6g}).")
            if _scalar_match(fa, t, rel)
            else (
                0.0,
                f"Expected ≈ {t:.6g}; not found. Check exact unit names and conversion direction.",
            )
        )

    if kind == "chain":
        t = task["ground_truth"]["final_value"]
        return (
            (1.0, f"Correct ({t:.6g}).")
            if _scalar_match(fa, t, rel)
            else (
                0.0,
                f"Expected final ≈ {t:.6g}. Do the 3 steps in order, threading each result into the next.",
            )
        )

    if kind == "aggregate":
        t = task["ground_truth"]["aggregate"]
        op = task["ground_truth"]["op"]
        return (
            (1.0, f"Correct {op}={t:.6g}.")
            if _scalar_match(fa, t, rel)
            else (
                0.0,
                f"Expected {op} ≈ {t:.6g}. Convert all to the common unit first, then aggregate.",
            )
        )

    if kind == "conditional":
        t = task["ground_truth"]["final_value"]
        return (
            (1.0, f"Correct ({t:.6g}).")
            if _scalar_match(fa, t, rel)
            else (0.0, f"Expected ≈ {t:.6g}. Compute step 1, pick the branch, then convert.")
        )

    # multi / multi_hard
    gt_items = task["ground_truth"]["items"]
    parsed: list[dict[str, Any]] = []
    m = re.search(r"\[.*\]", fa, re.S)
    if m:
        try:
            parsed = json.loads(m.group(0))
        except (json.JSONDecodeError, ValueError):
            parsed = []
    by_name = {str(p.get("name")): p for p in parsed if isinstance(p, dict)}
    correct, misses = 0, []
    for gi in gt_items:
        p = by_name.get(gi["name"])
        if not p:
            misses.append(f"{gi['name']}: missing")
            continue
        val_ok = any(
            _close(n, gi["converted_value"], rel)
            for n in _nums(json.dumps(p.get("converted_value")))
        )
        status_ok = str(p.get("status", "")).upper() == gi["status"]
        if val_ok and status_ok:
            correct += 1
        else:
            misses.append(
                f"{gi['name']}: exp {gi['converted_value']:.6g}/{gi['status']} got {p.get('converted_value')}/{p.get('status')}"
            )
    score = correct / len(gt_items) if gt_items else 0.0
    return score, f"{correct}/{len(gt_items)} correct." + (
        "" if not misses else " Issues: " + "; ".join(misses[:6])
    )
