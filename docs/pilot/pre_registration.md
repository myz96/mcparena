# mcparena Pilot — Pre-Registration

**Status:** READY FOR TAG. Server selection and task IDs are locked from the
pinned MCP-Bench commit (see below). The `locked_at_utc` timestamp is set when
this document is committed and `git tag -a pilot-prereg-v1` is run.

This document is the locked methodology for the mcparena pilot. It is
committed to the public repository and git-tagged (`pilot-prereg-v1`) **before
any pilot results exist**. Verify via `git log` that the tag predates any
`pilot-results/` artifacts on the same branch.

## Pilot question

Will any DSPy optimization mechanism meaningfully improve MCP server tool-use
success rates on the published MCP-Bench (Accenture, NeurIPS 2025) benchmark?

## Metric

**Mean task success rate per (server, condition).** Success is binary per
task: judged by the `Assess` Signature (see `src/mcparena/pilot/judge.py`)
using Claude Sonnet 4.6 as the same-model judge.

**Delta** for each non-baseline condition X: `delta = mean(X) − mean(baseline)`.

**Confidence interval:** 95% paired bootstrap CI via `scipy.stats.bootstrap`
with `n_resamples=1000`, `confidence_level=0.95`, `paired=True`,
`method="percentile"`.

## Conditions (5)

| # | Condition | Mechanism |
|---|---|---|
| 1 | `baseline` | Vanilla `dspy.ReAct` against MCP server, no optimization. |
| 2 | `miprov2` | `dspy.MIPROv2(auto="light")` over program signature. |
| 3 | `gepa` | `dspy.GEPA(auto="light")` via `gepa.adapters.mcp_adapter.MCPAdapter`. Reflection LM = same model (Claude Sonnet 4 via OpenRouter), `temperature=1.0`, `max_tokens=16000`. |
| 4 | `axis_ii` | Brute-force permutation search over the tool list (≤4 permutations per server). No optimizer; measures whether tool-list ordering alone changes outcomes. |
| 5 | `axis_iii` | Hand-inject a 1-shot example into each tool's description string. Re-evaluate. |

Rationale for five conditions: GEPA and MIPROv2 share a failure mode (both
optimize signature/prompt text); if mature MCP tool descriptions are already
well-tuned, both stagnate together. Axes (ii) and (iii) are mechanically
distinct levers that might catch what prompt optimizers miss.

## Task source

[Accenture/mcp-bench](https://github.com/Accenture/mcp-bench), NeurIPS 2025
Workshop on Scaling Environments for Agents. License: Apache 2.0.

**Pinned commit:** `7a8eaeae83a842a2949080acc5473f65e1569daf`

**Task field mapping** (from MCP-Bench JSON → `dspy.Example`):

| MCP-Bench field | `dspy.Example` field |
|---|---|
| `task_id` | `task_id` (traceability) |
| `task_description` | `user_request` (agent input) AND `expected_outcome` (judge ground truth — MCP-Bench has no separate criteria field; the description IS the criteria) |
| `fuzzy_description` | `mcp_bench_fuzzy` (preserved; not used in pilot) |
| `dependency_analysis`, `distraction_servers` | `mcp_bench_metadata` (preserved for audit) |

`expected_tools` (where present) is intentionally NOT mapped — pilot scores on
outcome, not on trajectory matching MCP-Bench's reference path.

## Servers (3, stratified by estimated difficulty)

MCP-Bench publishes only an overall leaderboard (claude-sonnet-4 ranks 5 at
0.681 overall); per-server scores are NOT published. Stratification here is
inferred from inspecting task complexity in the pinned commit.

| Tier | Server | `mcp_bench_id` | Task IDs |
|---|---|---|---|
| Easy | Math MCP | `Math MCP` | `math_mcp_000`, `math_mcp_001` |
| Medium | Wikipedia | `Wikipedia` | `wikipedia_000`, `wikipedia_001` |
| Hard | OpenAPI Explorer | `OpenAPI Explorer` | `openapi_explorer_000`, `openapi_explorer_001` |

Rationale:
- **Math MCP** (easy): deterministic numeric ops over small arrays; short
  trajectories; high expected baseline.
- **Wikipedia** (medium): search + extract + synthesize across articles;
  bounded multi-step.
- **OpenAPI Explorer** (hard): 5+ tool-call comparative audits across multiple
  API specs; biggest headroom for optimization.

All three are no-env-key, stdio-launchable from MCP-Bench's `mcp_servers/`
directory after `install.sh`.

## Sample size

3 servers × 2 tasks per server × 5 trials per condition × 5 conditions
= **150 trial-equivalents** (plus MIPROv2 and GEPA compile rollouts).

Per-server task counts are 2 each (MCP-Bench ships exactly 2 single-server
tasks per server). Lower statistical power than originally planned for; the
pilot is a feasibility / mechanism-validation spike, not a powered effect-size
study. Phase 1 will augment with multi-server tasks and/or MCP-Universe.

## Cost discipline

| Tier | Scope | Estimated cost |
|---|---|---|
| `--smoke-adapter` | 1 task, 1 trial, GEPA adapter load test | ~$1 |
| `--smoke-budget` | 1 server, 1 task, 2 trials, all 5 conditions | ~$8 |
| `--shake-out` | 1 server (Math MCP default), both tasks, 3 trials, all 5 conditions | ~$30 |
| Full pilot | 3 servers, all tasks, 5 trials, all 5 conditions | ~$60-80 |

Hard cap: **$300 cumulative**. Runner halts before exceeding. $50 reserved
for memo + retries; total wallet budget $350.

## Stopping rule

All planned trials run. **No early stopping.** The only halt is the cumulative
cost cap.

## Trial determinism

- Program / judge LM: `openrouter/anthropic/claude-sonnet-4`,
  `temperature=0.7`, `max_tokens=8192`
- Reflection LM (GEPA only): `openrouter/anthropic/claude-sonnet-4`,
  `temperature=1.0`, `max_tokens=16000`
- API routing: OpenRouter (single `OPENROUTER_API_KEY` for all providers).
  Matches MCP-Bench's published setup; routes to Anthropic for Claude calls.
- Model choice rationale: MCP-Bench's published baseline for Anthropic is
  "claude-sonnet-4" (Sonnet 4.0, score 0.681). Using the same model gives
  apples-to-apples comparison.
- No seed pinning across trials. Variance is intentional and integrated by
  the bootstrap CI.

## Decision criteria (after all conditions complete)

| Outcome | Interpretation | Next step |
|---|---|---|
| **PROCEED** | ANY of `{miprov2, gepa, axis_ii, axis_iii}` achieves both `CI low > 0` AND `point estimate ≥ 10 percentage points` on ≥2 of 3 servers | Begin full Builder plan: leaderboard URL, server-author outreach, GitHub Action, cross-model judging, MCP-Universe expansion. |
| **MIXED** | Strong signal on 1 server only, OR weak-but-positive signal (`CI low > 0`, point estimate 5-10pp) across all 3 | Narrower Builder: optimizer ships as research module; defer public leaderboard until mechanism replicates on richer benchmark. |
| **PIVOT** | No condition achieves `CI low > 0` on ≥2 servers | Drop the DSPy-optimization wedge. Re-scope mcparena around auto-task-generation, failure-taxonomy, or a different moat. |

## Reproducibility

- `git rev-parse pilot-prereg-v1` must equal the commit SHA on the branch that
  produces `pilot-results/`.
- `mcparena pilot` refuses to run if the working tree is dirty (override with
  `--allow-dirty` for smoke / shake-out runs only).
- All trial outputs saved to `pilot-results/` (gitignored; sanitized excerpts
  go into the public memo).
- The pilot memo published after results includes a verbatim copy of this
  pre-registration as an appendix.

## Methodology disclosures

- **Same-model judging:** Sonnet 4.6 judges its own outputs. MCP-Bench's own
  reported numbers use OpenAI o4-mini as the judge — direct comparison to
  their leaderboard is therefore approximate, not strict. The memo will
  report both our absolute scores AND the absolute differences vs MCP-Bench's
  published values for context, with the disclosure that the judges differ.
- **Cross-model judging deferred to Phase 1.**
- **Per-server baseline scores not published by MCP-Bench;** we measure our
  own baseline as the per-server reference.

## Out of scope for pilot (Phase 1 deliverables)

- Cross-model judging (Gemini judges Claude)
- 50-task gold-set judge calibration with ≥85% agreement
- Public leaderboard URL, server-author outreach, GitHub Action
- Additional benchmarks (MCP-Universe, MCPMark, MCPVerse)
- Per-server baseline replication of MCP-Bench's o4-mini-judged scores

## Lock

**Locked at UTC:** *set when `git tag -a pilot-prereg-v1` runs*
**Git tag:** `pilot-prereg-v1` (annotated)
**Verification:** `git show pilot-prereg-v1` must match this file's content
at the tagged commit.
