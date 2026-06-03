# mcparena pilot — launch memo

> **TL;DR:** GEPA self-discovered the Wikipedia MCP server's quirky tool argument names
> just by reading failed trajectories, and patched them into the agent prompt. Baseline
> 0% → GEPA 12.5% (n=16, +12.5pp, CI [0.0pp, +31.25pp]). Total cost: $6.29 on
> Qwen3-235b. We packaged the mechanism as `mcparena optimize <your-mcp-server>` so
> any MCP author can run the same diagnosis on their own server.

## The finding in one paragraph

Most LLMs hallucinate plausible-sounding kwargs (`topic=`, `max_facts=`) against
MCP tools whose real signatures use less obvious names (`topic_within_article=`,
`count=`). The base model's tool descriptions don't surface the parameter signature
clearly enough to prevent this. Across two Wikipedia tasks at n_trials=8 (n=16
paired evals), the baseline agent failed 100% of the time — every trajectory hit
`ValidationError: unexpected_keyword_argument` and never produced an answer. GEPA's
reflection LM read those failed trajectories, identified the schema mismatch, and
evolved the ReAct prompt to include:

> *"**Tool Limitations & Workarounds**: The `extract_key_facts` tool does not
> accept arguments named `topic`, `focus`, or `num_facts`. Instead, if the tool
> fails or its parameters are invalid, you must fall back to retrieving the full
> article using `get_article` and manually extract relevant facts..."*
>
> *"**Section Summarization Pitfall**: Always use the parameter name
> `section_title` (not `section`) when calling `summarize_article_section`.
> Otherwise, a validation error will occur."*

That single rewrite moved 2/16 trials from failure to success — a +12.5pp lift
with a 95% bootstrap CI of [0.0pp, +31.25pp]. The lift itself is borderline by
strict pre-registration criteria; the *mechanism* — an optimizer rediscovering
schema knowledge the base model couldn't infer — is what makes the finding
publishable.

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

- **n=1 viable server.** Math saturates at the ceiling (75% baseline, no
  headroom). OpenAPI Explorer infeasible at our model's 256K context (single
  tool response ~1.6M tokens). Only Wikipedia produced a measurable signal.
  Generalization claim is currently hand-wavy; second-server hunt is the
  immediate next step.
- **CI lower bound = exactly 0.0pp.** Bootstrap with two successes always
  includes resamples where neither is drawn. Strict pre-reg PROCEED gate not
  met; MIXED outcome by the locked decision criteria.
- **Same-model judging.** Qwen3-235b grades its own outputs. Cross-model
  judging is a Phase 1 deliverable.
- **Schema-quirk lift conflates two effects:** the new prompt both works
  around bad kwargs AND simplifies the ReAct scaffolding. We didn't ablate.
- **No comparison to a "fix tool descriptions upstream" baseline.** If you
  patch the Wikipedia MCP server's tool descriptions to surface the kwarg
  names clearly, the baseline might recover entirely — making GEPA's lift
  zero. That experiment is the next high-leverage validation.

## Pointers

- Repo: <https://github.com/myz96/mcparena>
- Pre-registration: [`docs/pilot/pre_registration.md`](pre_registration.md)
- Raw Phase 2 results: [`pilot-results/phase2.json`](../../pilot-results/phase2.json)
- CLI walkthrough: [README quickstart](../../README.md#quickstart)

— Michael Zhao, 2026-06-03
