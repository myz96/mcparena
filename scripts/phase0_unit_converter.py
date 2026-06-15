"""Phase 0: verifiable-task generator + verifier for the unit-converter MCP.

Backwards-from-execution ("the server is the oracle"):
  sample params -> EXECUTE against the server -> capture the real result as
  ground truth -> template a natural-language task -> verify the agent's answer
  programmatically (tolerance match). No LLM judge, no human answer key.

Two task kinds:
  - single: one conversion (likely easy -> saturation check)
  - multi : K conversions + per-item PASS/FAIL vs threshold (headroom; mirrors
            the real MCP-Bench unit_converter task shape)

This module is import-safe (generator + verifier are functions). Running it as
__main__ generates a small sample set and prints it for eyeballing — server
calls only, ~$0.

Usage:
    uv run python scripts/phase0_unit_converter.py            # eyeball samples
"""

from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path
from typing import Any

sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]

REPO_ROOT = Path(__file__).resolve().parents[1]
UC_CWD = str(REPO_ROOT / "third_party/mcp-bench-tasks/mcp_servers/unit-converter-mcp")
OUT_DIR = REPO_ROOT / "examples" / "verifiable"
TOLERANCE_REL = 1e-3

# Types whose value range is safe to sample positively in [1, 1000].
_SAFE_TYPES = [
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


def _stdio() -> Any:
    from mcp.client.stdio import StdioServerParameters

    return StdioServerParameters(
        command="uv", args=["run", "unit-converter-mcp"], env={}, cwd=UC_CWD
    )


def _supported_units(session: Any) -> dict[str, list[str]]:
    return json.loads(session.call_tool("list_supported_units"))


def _convert(session: Any, ctype: str, value: float, from_unit: str, to_unit: str) -> float:
    """Execute one conversion against the server; return the converted value (ground truth)."""
    raw = session.call_tool(f"convert_{ctype}", value=value, from_unit=from_unit, to_unit=to_unit)
    return float(json.loads(raw)["converted_value"])


def _sample_pair(rng: random.Random, units: list[str]) -> tuple[str, str]:
    a, b = rng.sample(units, 2)
    return a, b


def generate_tasks(
    session: Any,
    seed: int = 0,
    n_single: int = 4,
    n_multi: int = 3,
    k_range: tuple[int, int] = (4, 7),
) -> list[dict[str, Any]]:
    """Generate verifiable tasks by executing sampled conversions against the server."""
    rng = random.Random(seed)
    supported = _supported_units(session)
    types = [t for t in _SAFE_TYPES if t in supported and len(supported[t]) >= 2]
    tasks: list[dict[str, Any]] = []

    # --- single conversions ---
    for i in range(n_single):
        ctype = rng.choice(types)
        from_u, to_u = _sample_pair(rng, supported[ctype])
        value = round(rng.uniform(1, 1000), 2)
        gt = _convert(session, ctype, value, from_u, to_u)
        tasks.append(
            {
                "task_id": f"uc_single_{i:03d}",
                "kind": "single",
                "conversion_type": ctype,
                "user_request": (
                    f"Convert {value} {from_u} to {to_u} using the unit-converter tools. "
                    f"Respond with the numeric converted value."
                ),
                "ground_truth": {"converted_value": gt},
                "tolerance_rel": TOLERANCE_REL,
            }
        )

    # --- multi-step: K readings, each converted + PASS/FAIL vs a threshold ---
    for i in range(n_multi):
        k = rng.randint(*k_range)
        items = []
        for j in range(k):
            ctype = rng.choice(types)
            from_u, to_u = _sample_pair(rng, supported[ctype])
            value = round(rng.uniform(1, 1000), 2)
            converted = _convert(session, ctype, value, from_u, to_u)
            # threshold in target unit; factor <1 => PASS, >1 => FAIL (guarantees a mix)
            factor = rng.uniform(0.8, 1.2)
            threshold = round(converted * factor, 4)
            status = "PASS" if converted >= threshold else "FAIL"
            items.append(
                {
                    "name": f"sensor_{j + 1}",
                    "value": value,
                    "from_unit": from_u,
                    "to_unit": to_u,
                    "conversion_type": ctype,
                    "threshold": threshold,
                    "ground_truth_converted": converted,
                    "ground_truth_status": status,
                }
            )
        lines = "\n".join(
            f"  - {it['name']}: {it['value']} {it['from_unit']} -> {it['to_unit']} "
            f"(threshold {it['threshold']} {it['to_unit']})"
            for it in items
        )
        tasks.append(
            {
                "task_id": f"uc_multi_{i:03d}",
                "kind": "multi",
                "user_request": (
                    f"You are validating {k} sensor readings. For EACH reading: convert the value "
                    f"to its target unit using the unit-converter tools, then mark PASS if the "
                    f"converted value >= its threshold, else FAIL.\n\nReadings:\n{lines}\n\n"
                    f"Return a JSON array; one object per sensor with keys: name, converted_value, status."
                ),
                "ground_truth": {
                    "items": [
                        {
                            "name": it["name"],
                            "converted_value": it["ground_truth_converted"],
                            "status": it["ground_truth_status"],
                        }
                        for it in items
                    ]
                },
                "tolerance_rel": TOLERANCE_REL,
            }
        )
    return tasks


# ----------------------------- verifier -----------------------------


def _nums(text: str) -> list[float]:
    return [float(x) for x in re.findall(r"-?\d+\.?\d*(?:[eE][+-]?\d+)?", text or "")]


def _close(a: float, b: float, rel: float) -> bool:
    return abs(a - b) <= max(rel * abs(b), rel)


def verify(task: dict[str, Any], final_answer: str) -> tuple[float, str]:
    """Programmatic check → (score in [0,1], feedback string for GEPA)."""
    rel = task.get("tolerance_rel", TOLERANCE_REL)
    fa = final_answer or ""
    if task["kind"] == "single":
        target = task["ground_truth"]["converted_value"]
        if any(_close(n, target, rel) for n in _nums(fa)):
            return 1.0, f"Correct: found {target:.6g} in the answer."
        return 0.0, (
            f"Incorrect: expected converted value ≈ {target:.6g}, not present in answer. "
            f"Check the tool's required arg names and that you converted in the right direction."
        )
    # multi
    gt_items = task["ground_truth"]["items"]
    parsed: list[dict[str, Any]] = []
    try:
        m = re.search(r"\[.*\]", fa, re.S)
        if m:
            parsed = json.loads(m.group(0))
    except (json.JSONDecodeError, ValueError):
        parsed = []
    by_name = {str(p.get("name")): p for p in parsed if isinstance(p, dict)}
    correct = 0
    misses: list[str] = []
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
                f"{gi['name']}: expected {gi['converted_value']:.6g}/{gi['status']}, "
                f"got {p.get('converted_value')}/{p.get('status')}"
            )
    score = correct / len(gt_items) if gt_items else 0.0
    fb = f"{correct}/{len(gt_items)} sensors correct." + (
        "" if not misses else " Issues: " + "; ".join(misses[:6])
    )
    return score, fb


def main() -> int:
    from mcparena.pilot import tools

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with tools.persistent_session(_stdio()) as session:
        tasks = generate_tasks(session, seed=0, n_single=4, n_multi=3)

    out_path = OUT_DIR / "unit_converter_tasks.json"
    out_path.write_text(json.dumps(tasks, indent=2))

    print(f"\n=== generated {len(tasks)} verifiable tasks → {out_path} ===\n")
    for t in tasks:
        print(f"[{t['task_id']}] kind={t['kind']}")
        req = t["user_request"]
        print("  request:", (req[:300] + ("…" if len(req) > 300 else "")).replace("\n", "\n  "))
        print("  ground_truth:", json.dumps(t["ground_truth"])[:400])
        print()

    # quick self-test of the verifier on a correct + wrong answer for task 0
    t0 = tasks[0]
    gt0 = t0["ground_truth"]["converted_value"]
    print("=== verifier self-test (single task 0) ===")
    print("  correct answer →", verify(t0, f"The result is {gt0}."))
    print("  wrong answer   →", verify(t0, "The result is 0.123."))
    return 0


if __name__ == "__main__":
    sys.exit(main())
