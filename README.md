# mcparena

> The optimization layer over MCP benchmarks. Improve your MCP server's score on [MCP-Bench](https://github.com/Accenture/mcp-bench) via [GEPA](https://github.com/gepa-ai/gepa) + [DSPy](https://github.com/stanfordnlp/dspy).

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![Status](https://img.shields.io/badge/Status-Alpha-orange.svg)](#status)

## Status

**Alpha — pilot phase, public for visibility.** This is a pre-registered, in-progress spike validating whether DSPy optimizers (GEPA, MIPROv2) and tool-list mechanisms (ordering permutations, 1-shot example injection) can meaningfully improve MCP server tool-use success rates on the [MCP-Bench](https://github.com/Accenture/mcp-bench) benchmark.

### Pilot results so far

| Server | Baseline (vanilla `dspy.ReAct`) | GEPA | Δ | 95% CI | Cost |
|---|---|---|---|---|---|
| Math MCP (easy) | 75% (3/4) | 75% (matched) | 0pp — saturated, no headroom | — | — |
| Wikipedia (medium) | 0% (0/16) | 12.5% (2/16) | +12.5pp | [0.0pp, +31.25pp] | $6.29 |
| **Time MCP (medium)** | **0% (0/8)** | **37.5% (3/8)** | **+37.5pp** | **[+12.5pp, +75.0pp]** | **$0.07** |
| OpenAPI Explorer (hard) | infeasible | — | single tool response = 1.6M tokens, exceeds 256K context | — | — |

**Two servers show real GEPA lift.** Time MCP strictly meets the pre-registered PROCEED gate (`CI_low > 0 AND delta ≥ 10pp`); Wikipedia meets the non-strict reading. The pre-reg required ≥2 of 3 servers — generous read: gate met; strict read: 1.5/3.

The cost spread is its own finding: Wikipedia trajectories burn $6 because every tool call hits the Wikipedia API and 5+ rate-limit waits per run consume LLM tokens while idle. Time MCP tools run locally; same n × paired analysis costs **90× less**.

### What GEPA actually discovered

**Wikipedia: GEPA patched the model's wrong-kwarg hallucinations.** Baseline trajectories failed because the model called `extract_key_facts(topic=..., max_facts=5)` against a tool whose real signature is `extract_key_facts(title, topic_within_article, count)` — 77+ `ValidationError: unexpected_keyword_argument` failures across the run. The tool descriptions don't surface the kwarg signature. GEPA's reflection LM read the failed trajectories and rewrote the ReAct prompt to include:

> *"**Tool Limitations & Workarounds**: The `extract_key_facts` tool does not accept arguments named `topic`, `focus`, or `num_facts`. Instead, if the tool fails or its parameters are invalid, you must fall back to retrieving the full article using `get_article` and manually extract relevant facts..."*
>
> *"**Section Summarization Pitfall**: Always use the parameter name `section_title` (not `section`) when calling `summarize_article_section`. Otherwise, a validation error will occur."*

**Time MCP: GEPA discovered procedural structure, not schema quirks.** Same model, same optimizer, same harness — but a different bottleneck. Time MCP's two tasks need an iterative slot-search algorithm with state tracked across many tool calls, and baseline trajectories ran out of `max_iters` before assembling a correct JSON answer. GEPA's reflection rewrote the prompt to encode the algorithm:

> *"1. **Initialize Context**: Always begin by retrieving the current time... 2. **Use Tools Correctly**... 3. **Handle Tool Failures Gracefully**... 4. **Simulate Conversions if Tools Fail Persistently**: As a fallback, use observed UTC offsets... 5. **Iterative Evaluation of Candidate Slots**: For each candidate: convert... check business hours... increment... wrap to next day at 09:00..."*

**The point is the same mechanism, applied to whatever the actual bottleneck is.** Wikipedia needed kwarg knowledge; Time MCP needed procedural scaffolding. GEPA observed the failure mode and patched it — that's the pitch. The discovered prompt is a deployable artifact: an MCP server author can ship it as the recommended system prompt and any agent using their server inherits the optimizer's discoveries.

The full vision — public leaderboard, server-author outreach, GitHub Action CI gate — ships once the schema-and-procedural-discovery pattern is shown to generalize across more server types.

## Project ethos

- **Pre-registration discipline.** Metric, decision criteria, and MCP-Bench task IDs are locked in [`docs/pilot/pre_registration.md`](docs/pilot/pre_registration.md), git-tagged `pilot-prereg-v2` *before* any results exist. Verify via `git log`.
- **Build on existing benchmarks.** mcparena uses [MCP-Bench (Accenture, NeurIPS 2025)](https://github.com/Accenture/mcp-bench) as ground truth rather than reinventing measurement. Future phases will integrate MCP-Universe, MCPMark, MCPVerse.
- **Multiple mechanisms.** Pilot tests five conditions: baseline / MIPROv2 / GEPA / tool ordering (axis ii) / 1-shot example injection (axis iii). Strong soft-fail mitigation — if one stagnates, others may lift.
- **Per-condition checkpointing.** Mid-run crashes (process kill, rate limit, network) lose at most the in-flight condition; completed conditions resume from `pilot-results/{mode}.json` on relaunch.

## Quickstart

```bash
# Clone + install
git clone https://github.com/myz96/mcparena.git
cd mcparena
uv sync

# Required: OpenRouter API key (one key covers all providers used in pilot)
cp .env.example .env  # then edit .env to set OPENROUTER_API_KEY=sk-or-...

# Smoke test the GEPA MCP adapter — verifies install (~$0)
mcparena pilot --smoke-adapter

# Calibrate cost on one server (1 server × 1 task × 2 trials × 5 conditions, ~$0.20)
mcparena pilot --smoke-budget

# Shake-out gate (1 server × all tasks × 3 trials × 5 conditions)
mcparena pilot --shake-out --server wikipedia

# Full pilot (3 servers × all tasks × 5 trials × 5 conditions)
mcparena pilot

# Optimize ANY MCP server — point at a command + task file, get back the
# GEPA-discovered prompt (any tool-schema quirks GEPA found) + delta vs baseline
mcparena optimize \
  --server-cmd "uv run python -m wikipedia_mcp" \
  --tasks examples/tasks_wikipedia.json \
  --n-trials 3 \
  --output-dir my-results/
```

`mcparena optimize` writes `results.json` (raw scores + cost + discovered prompt)
and `summary.md` (human-readable table + a fenced block of the prompt GEPA
evolved). When the model has been hallucinating wrong tool kwargs against your
server, the discovered prompt will typically contain a `### Critical Task
Constraints` section listing them — fix those upstream in your tool descriptions
and the lift may shrink to zero (a positive signal: your descriptions are now
complete enough that the base model gets it right without GEPA).

All runs write per-condition checkpoints to `pilot-results/{mode}.json`. Re-running the same command skips already-completed conditions.

## Architecture

| Layer | Tool |
|---|---|
| Program / judge LM | `qwen/qwen3-235b-a22b-2507` via OpenRouter (MCP-Bench rank 6, score 0.678 — essentially tied with claude-sonnet-4 at 0.681 within margin of error; ~150× cheaper output tokens) |
| GEPA reflection LM | Same model, higher temperature + larger window (`temperature=1.0`, `max_tokens=16000`) |
| Optimizers | DSPy [MIPROv2](https://arxiv.org/abs/2406.11695) + [GEPA](https://arxiv.org/abs/2507.19457) (native MCP adapter) |
| Task source | [MCP-Bench](https://github.com/Accenture/mcp-bench) pinned at commit `7a8eaeae…` |
| MCP transport | stdio (pilot); HTTP/SSE planned for Phase 1 |
| Statistical aggregation | `scipy.stats.bootstrap` (paired 95% CI, `n_resamples=1000`, percentile method) |
| Cost discipline | Hard cap $300 cumulative (pre-reg R6); per-condition budget tracked in `costs.py` |

## Docs

- [`docs/pilot/pre_registration.md`](docs/pilot/pre_registration.md) — locked pilot metric, decision criteria, model + cost amendment trail
- Cleaned design + planning docs are pending — raw transcripts live in `docs/_drafts/` (gitignored).

## License

[MIT](LICENSE) — third-party servers in `third_party/` (gitignored) retain their upstream licenses (mostly Apache 2.0 / MIT). MCP-Bench task IDs and JSON schemas are used under [Apache 2.0](https://github.com/Accenture/mcp-bench/blob/main/LICENSE).
