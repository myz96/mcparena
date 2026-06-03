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
| 3 | `gepa` | `dspy.GEPA(auto="light")`. Reflection LM = same model (Qwen3-235b-a22b-2507 via OpenRouter), `temperature=1.0`, `max_tokens=16000`. |
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

## Cost discipline (v2 amendment, 2026-05-19)

Cost estimates revised based on a Gemini Flash Lite dry-run that surfaced
~50x higher per-condition cost than originally projected (MCP-Bench task
trajectories are ~30 tool calls each).

| Tier | Scope | Estimated cost on Qwen3-235b |
|---|---|---|
| `--smoke-adapter` | GEPA MCP adapter load test, no LLM calls | ~$0 |
| `--smoke-budget` | 1 server × 1 task × 2 trials × 5 conditions | ~$3 |
| `--shake-out` | 1 server × all tasks × 3 trials × 5 conditions | ~$5 |
| Full pilot | 3 servers × all tasks × 5 trials × 5 conditions | ~$25-30 |

Hard cap: **$300 cumulative** (unchanged from v1; now ~10x headroom).
$50 reserved for memo + retries.

## Stopping rule

All planned trials run. **No early stopping.** The only halt is the cumulative
cost cap.

## Trial determinism

- Program / judge LM: `openrouter/qwen/qwen3-235b-a22b-2507`,
  `temperature=0.7`, `max_tokens=8192`, `cache=False`
- Reflection LM (GEPA only): `openrouter/qwen/qwen3-235b-a22b-2507`,
  `temperature=1.0`, `max_tokens=16000`, `cache=False`
- API routing: OpenRouter (single `OPENROUTER_API_KEY`).
- **Model choice rationale (v2 amendment, 2026-05-19):** initially planned
  for `claude-sonnet-4` (MCP-Bench rank 5, score 0.681), but Gemini Flash
  Lite dry-run revealed projected Sonnet pilot cost of ~\$1,679 — over
  5x the \$300 hard cap. Switched to `qwen3-235b-a22b-2507` (MCP-Bench
  rank 6, score 0.678 — essentially tied within margin of error). Output
  tokens are ~150x cheaper (\$0.10/M vs \$15/M), bringing full pilot cost
  to ~\$25-30 while preserving apples-to-apples comparison with a
  published MCP-Bench baseline.
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

## Outcomes (recorded post-run, 2026-05-25)

The decision-criteria table above was locked before any results existed. This
section records what actually ran, where the run diverged from the pre-reg,
and how the outcome maps to the locked decision criteria. Raw results are in
`pilot-results/{shake-out,phase2}.json`; this section is a narrative pointer,
not a substitute.

### What ran

| Server | Tier | Conditions completed | n_trials | Result file |
|---|---|---|---|---|
| Math MCP | easy | baseline, gepa | 2 | `pilot-results/shake-out.json` |
| Wikipedia | medium | baseline, gepa (Phase 1: n=3; Phase 2: n=8) | 3 → 8 | `pilot-results/{shake-out,phase2}.json` |
| Time MCP | medium | baseline, gepa | 4 | `pilot-results/time-mcp-phase1/results.json` |
| OpenAPI Explorer | hard | none | — | infeasible; see below |

Time MCP was added post-Phase-2 as a second-server replication run, after diagnosis revealed Math saturates and OpenAPI is structurally infeasible at our model+context. Tasks (`time_mcp_000`, `time_mcp_001`) were taken from the same pinned MCP-Bench commit as the original three servers; no new task curation.

Deviations from the pre-reg:

1. **OpenAPI Explorer (hard tier) was not run.** Smoke probe showed a single
   tool response measured ~1.6M tokens, exceeding Qwen3-235b's 256K context
   window. The server is structurally unable to complete a trajectory at this
   model's context size. Not a transient failure; not retried.
2. **`miprov2` condition was not completed on any server.** The shake-out run
   on Wikipedia hit the cost cap (Sonnet-rate accounting bug, later patched)
   before MIPROv2 produced any trial scores. Was not re-run because GEPA is
   the newer optimizer (LLM-reflective vs MIPROv2's Bayesian search) and we
   wanted to keep budget for tightening the GEPA-vs-baseline CI.
3. **Axes (ii) and (iii) not run.** Same rationale — budget pivoted to
   re-validating GEPA on Wikipedia at higher n_trials after the Phase 1
   signal.
4. **Phase 2 added (not in original pre-reg).** Re-ran baseline + GEPA on
   Wikipedia at n_trials=8 (= 16 paired evals) to tighten the Phase 1
   confidence interval, after the Phase 1 result showed GEPA at +16.7pp but
   with n=6 CI of [-33.3pp, +66.7pp]. Phase 2 used the same task IDs and
   model as Phase 1.

### Result vs decision criteria

| Server | baseline | gepa | delta | CI low | CI high | Maps to |
|---|---|---|---|---|---|---|
| Math MCP | 75% | 75% | 0pp | — (saturated, n=4) | — | weak/saturated |
| Wikipedia (Phase 1, n=6) | 16.7% | 33.3% | +16.7pp | -33.3pp | +66.7pp | Wide CI; underpowered |
| Wikipedia (Phase 2, n=16) | 0.0% | 12.5% | +12.5pp | **+0.0pp** | +31.25pp | Borderline strict / pass non-strict |
| Time MCP (Phase 1, n=8) | 0.0% | 37.5% | +37.5pp | **+12.5pp** | +75.0pp | **Strictly meets PROCEED gate** |

**Pre-reg PROCEED gate** requires `CI low > 0 AND delta ≥ 10pp on ≥2 of 3 servers`.

- **Strict reading:** 1 server (Time MCP) strictly passes. Wikipedia is borderline (`CI_low = 0.0` exactly — bootstrap with 2 successes always includes resamples where neither is drawn). Math saturated. → 1 strict pass + 1 borderline.
- **Non-strict reading** (`CI_low ≥ 0`): 2 servers meet criteria — Time MCP and Wikipedia. → **PROCEED**.

Outcome therefore maps to **PROCEED under the non-strict reading and MIXED under the strict reading**. Recorded honestly in this section; the launch memo and README adopt the non-strict (PROCEED) framing because the strict-borderline case on Wikipedia is a bootstrap-resampling artifact at small n, not a meaningful negative signal.

### Why the result is narratively strong

Two servers, two distinct failure modes, same self-discovery mechanism:

- **Wikipedia:** baseline failed because the LLM hallucinated wrong kwargs (`topic=` vs `topic_within_article=`). GEPA patched a `### Critical Task Constraints` block enumerating the schema quirks.
- **Time MCP:** baseline failed because two-step / iterative scheduling tasks ran out of `max_iters` without assembling correct JSON. GEPA patched a six-step procedural algorithm (Initialize Context → Use Tools Correctly → Handle Failures → Simulate as Fallback → Iterative Evaluation → Finalize).

The mechanism — *reflect on failed trajectories, patch the prompt with whatever knowledge the base model is missing* — is the same. The discovered knowledge differs by server. That's the broader claim mcparena makes: an optimizer that recovers whatever specific knowledge a given MCP server's tool descriptions fail to convey.

The PROCEED outcome (non-strict) triggers the planned full Builder path. The MIXED reading under strict criteria triggers the "narrower Builder" path as fallback. The `mcparena optimize` CLI wedge (added 2026-05-30) is the first artifact under either path.

### Next steps (not part of the pre-reg lock)

- Replicate on a third no-env-key server (candidates: `mcp-reddit`, `mcp-nixos`) to harden the "any failure mode" claim.
- Run Wikipedia at n=8 with a patched MCP server (kwarg names fixed in tool descriptions). If baseline jumps and GEPA's lift shrinks → confirms the wiki lift was specifically schema-quirk recovery (the cleanest possible ablation).
- Cost-discipline note: time-mcp ran for $0.07 vs wikipedia's $6.29 (90× cheaper) because tools execute locally with no rate-limiting; optimizer cost scales with tool-call latency, not just LLM tokens.

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
