# mcparena Pilot — Pre-Registration

**Status:** DRAFT — server selection and task IDs to be locked during Day-1 sub-sequence (clone MCP-Bench → review leaderboard → pick 3 stratified servers → finalize this document → `git tag -a pilot-prereg-v1`).

This document is the locked methodology for the mcparena pilot. It is committed to the public repository and git-tagged (`pilot-prereg-v1`) **before any pilot results exist**. Verify via `git log` that the tag predates the `pilot-results/` artifacts in the same branch.

## Pilot question

Will any DSPy optimization mechanism meaningfully improve MCP server tool-use success rates on the published MCP-Bench (Accenture, NeurIPS 2025) benchmark?

## Metric

**Mean task success rate per (server, condition).** Success is binary per task: judged by the `Assess` Signature (see `src/mcparena/pilot/judge.py`) using Claude Sonnet 4.6 as the same-model judge.

**Delta** for each non-baseline condition X: `delta = mean(X) − mean(baseline)`.

**Confidence interval:** 95% paired bootstrap CI via `scipy.stats.bootstrap` with `n_resamples=1000`, `confidence_level=0.95`, `paired=True`, `method="percentile"`.

## Conditions (5)

| # | Condition | Mechanism |
|---|---|---|
| 1 | `baseline` | Vanilla `dspy.ReAct` against MCP server, no optimization. |
| 2 | `miprov2` | `dspy.MIPROv2(auto="light")` over program signature. |
| 3 | `gepa` | `dspy.GEPA(auto="light")` with reflection LM Claude Opus 4.7 (`max_tokens=16000`, `temperature=1.0`). |
| 4 | `axis_ii` | Brute-force permutation search over the tool list (≤4 permutations per server). No optimizer; measures whether tool-list ordering alone changes outcomes. |
| 5 | `axis_iii` | Hand-inject a worked 1-shot example into each tool's description string. Re-evaluate. |

Rationale for five conditions: GEPA and MIPROv2 share a failure mode (both optimize signature/prompt text via Bayesian or reflective methods); if mature MCP tool descriptions are already well-tuned, both stagnate together. Axes (ii) and (iii) are mechanically distinct levers that might catch what prompt optimizers miss. Cost overhead ~$20.

## Task source

[Accenture/mcp-bench](https://github.com/Accenture/mcp-bench), NeurIPS 2025 Workshop on Scaling Environments for Agents.

**Pinned commit:** `<TBD — set during Day-1 sub-sequence>`

**Task field mapping** (from MCP-Bench JSON → `dspy.Example`):

| MCP-Bench field | `dspy.Example` field | Purpose |
|---|---|---|
| `task_id` | `task_id` | Traceability |
| `user_query` | `user_request` | Agent input |
| `success_criteria` (collapsed to "Task succeeds if ALL of: (a)..., (b)...") | `expected_outcome` | Judge ground truth |
| (entire record) | `mcp_bench_rubric` | Preserved for audit |

`expected_tools` is intentionally NOT mapped — pilot scores on outcome (did the agent achieve the goal?) not on trajectory matching MCP-Bench's reference path.

## Servers (3, stratified by published baseline)

To be selected during Day-1 sub-sequence by reviewing the MCP-Bench leaderboard for Claude Sonnet 4.6's per-server baseline scores, then picking one each across difficulty tiers and filtered to stdio-launchable on the development machine.

| Tier | Target baseline (Sonnet 4.6) | Server | MCP-Bench task IDs |
|---|---|---|---|
| Easy | ~70-90% (low headroom; ceiling test) | `<TBD>` | `<TBD>` |
| Medium | ~40-60% (workhorse; medium headroom) | `<TBD>` | `<TBD>` |
| Hard | <30% (most headroom; cherry of optimization) | `<TBD>` | `<TBD>` |

## Sample size

3 servers × ~10 tasks per server × 5 trials per condition × 5 conditions ≈ **750 trial-equivalents** (plus MIPROv2 and GEPA compile rollouts in addition). Per-server task counts pinned in the server table above.

## Stopping rule

All planned trials run. **No early stopping.** The only halt is the cumulative cost cap of $300 (with $50 buffer reserved for memo + retries; total budget $350).

## Trial determinism

- Program LM: `anthropic/claude-sonnet-4-6`, `temperature=0.7`, `max_tokens=8192`
- Reflection LM (GEPA only): `anthropic/claude-opus-4-7`, `temperature=1.0`, `max_tokens=16000`
- No seed pinning across trials. Variance across trials is intentional and is what the bootstrap CI integrates over.

## Decision criteria

After all conditions complete and per-(server, condition) CIs are computed:

| Outcome | Interpretation | Next step |
|---|---|---|
| **PROCEED** | ANY of `{miprov2, gepa, axis_ii, axis_iii}` achieves both `CI low > 0` AND `point estimate ≥ 10 percentage points` on ≥2 of 3 servers | Begin full Builder plan: leaderboard URL, server-author outreach, GitHub Action, multi-benchmark coverage, cross-model judging. |
| **MIXED** | Strong signal on 1 server only, OR weak-but-positive signal (`CI low > 0`, point estimate 5-10pp) across all 3 | Narrower Builder: optimizer ships as research module; defer leaderboard until mechanism replicates. |
| **PIVOT** | No condition achieves `CI low > 0` on ≥2 servers | Drop the DSPy-optimization wedge. Re-scope mcparena around auto-task-generation, failure-taxonomy, or a different moat. |

## Reproducibility

- `git rev-parse pilot-prereg-v1` must equal the commit SHA on the branch that produces `pilot-results/`.
- `mcparena pilot` refuses to run if the working tree is dirty (override with `--allow-dirty` for smoke runs only).
- All trial outputs saved to `pilot-results/` (gitignored; sanitized excerpts go into the public memo).
- The pilot memo published after results includes a verbatim copy of this pre-registration as an appendix.

## Out of scope for pilot

Per plan v5.1 separation of pilot vs full Builder:

- Cross-model judging (Phase 1 deliverable; pilot uses same-model)
- 50-task gold-set judge calibration with ≥85% agreement (Phase 1; pilot ships descriptive 10-transcript fixture)
- Public leaderboard URL, server-author outreach, GitHub Action — all Phase 1
- Additional benchmarks (MCP-Universe, MCPMark, MCPVerse) — Phase 1 expansion

## Lock

**This document is locked at:** `<TBD — UTC timestamp set during Day-1 sub-sequence>`
**Git tag:** `pilot-prereg-v1` (annotated; `git tag -a pilot-prereg-v1 -m "Pilot pre-registration"`)
**Verification:** `git show pilot-prereg-v1` must match this file's content at the tagged commit.
