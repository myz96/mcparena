# mcparena pilot — launch memo

> **TL;DR:** GEPA self-discovered different bottlenecks on two different MCP servers — wrong-kwarg hallucinations on Wikipedia, missing procedural structure on Time MCP — and patched both by reflecting on failed trajectories. Pre-registered PROCEED gate is met (non-strict reading) across 2 of 3 servers. The mechanism ships as `mcparena optimize <your-mcp-server>` so any MCP author can run the same diagnosis on their own server.

| Server | baseline | GEPA | Δ | 95% CI | cost |
|---|---|---|---|---|---|
| Wikipedia | 0.0% (0/16) | 12.5% (2/16) | +12.5pp | [0.0pp, +31.25pp] | $6.29 |
| Time MCP | 0.0% (0/8) | **37.5% (3/8)** | **+37.5pp** | **[+12.5pp, +75.0pp]** | **$0.07** |
| Math MCP | 75% | 75% | 0pp — saturated | — | — |

## The finding — same mechanism, two different failure modes

**Wikipedia: GEPA patched wrong-kwarg hallucinations.** The base model called
`extract_key_facts(topic=..., max_facts=5)` against a tool whose real signature
is `extract_key_facts(title, topic_within_article, count)` — 77+ `ValidationError`
failures across the run because tool descriptions don't surface the kwarg signature
clearly. GEPA's reflection LM rewrote the ReAct prompt to include:

> *"**Tool Limitations & Workarounds**: The `extract_key_facts` tool does not
> accept arguments named `topic`, `focus`, or `num_facts`. Instead, if the tool
> fails or its parameters are invalid, you must fall back to retrieving the full
> article using `get_article` and manually extract relevant facts..."*
>
> *"**Section Summarization Pitfall**: Always use the parameter name
> `section_title` (not `section`) when calling `summarize_article_section`.
> Otherwise, a validation error will occur."*

**Time MCP: GEPA patched missing procedural structure.** Same model, same harness,
same optimizer — different bottleneck. Time MCP's scheduling tasks need an
iterative slot-search algorithm with state tracked across many tool calls;
baseline ran out of `max_iters` before assembling correct JSON. GEPA discovered
the algorithm and wrote it down:

> *"1. **Initialize Context**: Always begin by retrieving the current time in at
> least one relevant timezone... 2. **Use Tools Correctly**... 3. **Handle Tool
> Failures Gracefully**... 4. **Simulate Conversions if Tools Fail Persistently**:
> As a fallback, use observed UTC offsets... 5. **Iterative Evaluation of
> Candidate Slots**: For each candidate: convert... check business hours...
> increment... If past 17:00, wrap to next day at 09:00..."*

The lifts (12.5pp Wikipedia, 37.5pp Time MCP) are side-effects. The point is
that **the same optimization mechanism patches whatever bottleneck a given MCP
server's tool descriptions and trajectories actually reveal** — kwarg knowledge
in one case, procedural scaffolding in another. The discovered prompt is a
deployable artifact: an MCP server author can ship it as the recommended
system prompt and any agent using their server inherits the discoveries.

## Why this matters beyond Wikipedia

MCP server tool descriptions are notoriously thin (`"Extract key facts from a
Wikipedia article, optionally focused on a topic."`) and the kwarg signature is
buried in the JSON schema, not the description prose. The LLM's prior is to
guess conventional names; when those don't match, every call fails silently
until the trajectory exhausts its budget. This affects every MCP server, not
just Wikipedia — it just doesn't get measured because most MCP evaluations stop
at "did the server respond" rather than "did the agent succeed at the task."

GEPA's reflection-on-failure loop is structurally well-suited to fixing this
class of bug at the prompt layer, without requiring server-side changes. The
optimizer essentially does what a human integrator would do: read the errors,
figure out what the server actually wants, write it down.

## What we built

- **`mcparena optimize <server-cmd> --tasks tasks.json`** — runs baseline +
  GEPA against any MCP server, outputs `summary.md` with the discovered
  prompt + a paired bootstrap CI. ~$1-10 per run on Qwen3-235b depending on
  task complexity.
- **`docs/pilot/pre_registration.md`** — methodology locked at git tag
  `pilot-prereg-v2` *before* any results existed. Outcomes section appended
  post-run, decisions mapped honestly to PROCEED/MIXED/PIVOT criteria.
- **`pilot-results/{shake-out,phase2}.json`** — raw per-trial scores, costs,
  per-condition checkpoints. Reproducible from a clean clone.

## What we're asking

- **MCP server authors:** run `mcparena optimize` against your own server with
  3-10 representative tasks. If GEPA discovers a schema quirk, that's a
  candidate upstream fix — update your tool descriptions, run again, and the
  lift should shrink (a positive signal). Report the result either way; we'd
  like to characterize how common this pattern is across the ecosystem.
- **Optimizer researchers:** the discovered-prompt artifact is concrete
  reflection-LM output; suggestions on how to score / select between competing
  rewrites are welcome.
- **Anyone interested in agent eval:** the pilot pre-registration + outcomes
  doc is a small worked example of pre-registering an agent experiment with
  honest mapping to pre-locked decision criteria. Critiques welcome.

## Honest limitations

- **2 of 3 servers viable, third (OpenAPI Explorer) structurally infeasible
  at our model+context** (single `getApiOperation` response runs into the
  megabytes against the OpenAI/GitHub specs; trajectory exceeds Qwen3-235b's
  256K context window). Replication on more MCP servers is the immediate
  next step.
- **Wikipedia CI lower bound = exactly 0.0pp** (bootstrap with 2 successes
  always includes resamples where neither is drawn). Strict pre-reg gate not
  met on that server alone; non-strict gate is met. Time MCP's CI is strictly
  above zero, so the combined ≥2-of-3 PROCEED criterion holds under the
  non-strict reading.
- **Same-model judging.** Qwen3-235b grades its own outputs. Cross-model
  judging is a Phase 1 deliverable.
- **Discovered prompts conflate multiple effects.** Each evolves both
  schema/procedural hints AND simplifies the ReAct scaffolding. We didn't
  ablate which sub-effect carries which fraction of the lift.
- **No comparison to a "fix tool descriptions upstream" baseline.** If you
  patch the Wikipedia MCP server's tool descriptions to surface the kwarg
  names clearly, the baseline might recover entirely — making GEPA's lift
  zero. That experiment (Wiki) and the analogous "make trajectories explicit
  in the tool descriptions" experiment (Time MCP) are the next high-leverage
  validations.
- **GEPA's Time MCP prompt contains wrong kwarg names** for `convert_time`
  (`from_timezone`/`to_timezone`/`datetime` vs the actual schema's
  `source_timezone`/`time`/`target_timezone`). The procedural structure was
  load-bearing enough to win 3/8 anyway. With correct kwarg hints the lift
  would presumably be larger.

## Pointers

- Repo: <https://github.com/myz96/mcparena>
- Pre-registration: [`docs/pilot/pre_registration.md`](pre_registration.md)
- Raw Phase 2 (Wikipedia) results: [`pilot-results/phase2.json`](../../pilot-results/phase2.json)
- Raw Time MCP Phase 1 results: [`pilot-results/time-mcp-phase1/`](../../pilot-results/time-mcp-phase1/)
- CLI walkthrough: [README quickstart](../../README.md#quickstart)

— Michael Zhao, 2026-06-03
