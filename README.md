# mcparena

> The optimization layer over MCP benchmarks. Improve your MCP server's score on [MCP-Bench](https://github.com/Accenture/mcp-bench) via [GEPA](https://github.com/gepa-ai/gepa) + [DSPy](https://github.com/stanfordnlp/dspy).

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![Status](https://img.shields.io/badge/Status-Alpha-orange.svg)](#status)

## Status

**Alpha — pilot phase, public for visibility.** This is a pre-registered, in-progress spike validating whether DSPy optimizers (GEPA, MIPROv2) and tool-list mechanisms (ordering permutations, 1-shot example injection) can meaningfully improve MCP server tool-use success rates on the [MCP-Bench](https://github.com/Accenture/mcp-bench) benchmark.

### Pilot results so far

| Server | Baseline (vanilla `dspy.ReAct`) | GEPA | Δ | 95% CI |
|---|---|---|---|---|
| Math MCP (easy) | 75% (3/4) | 75% (matched) | 0pp — saturated, no headroom | — |
| Wikipedia (medium) | 0% (0/16) | **12.5% (2/16)** | **+12.5pp** | [0.0pp, +31.25pp] |
| OpenAPI Explorer (hard) | infeasible | — | single tool response = 1.6M tokens, exceeds 256K context | — |

Phase 2 on Wikipedia (n=16 paired evals at n_trials=8 per task × 2 tasks) cost $6.29 on Qwen3-235b. The CI lower bound clamps at exactly 0.0pp — strict pre-registration PROCEED gate (`CI_low > 0`) is *not* met; non-strict (`CI_low ≥ 0`) is. The pilot finding is therefore reported honestly as **borderline statistical, strong diagnostic** (see next section).

### What GEPA actually discovered

Wikipedia baseline at 0% looked like a regression from Phase 1's 1/6 (n=6 was too small to see the true rate). Investigation found the LLM was hallucinating wrong kwarg names against the wikipedia MCP server (e.g. `topic=` instead of `topic_within_article=` for `extract_key_facts`; `section=` instead of `section_title=` for `summarize_article_section`) — 77+ `ValidationError: unexpected_keyword_argument` failures across the run. Tool descriptions don't surface the kwarg signature.

GEPA's reflection LM read the failed trajectories and rewrote the ReAct prompt to include an explicit constraints block:

> *"**Tool Limitations & Workarounds**: The `extract_key_facts` tool does not accept arguments named `topic`, `focus`, or `num_facts`. Instead, if the tool fails or its parameters are invalid, you must fall back to retrieving the full article using `get_article` and manually extract relevant facts..."*
>
> *"**Section Summarization Pitfall**: Always use the parameter name `section_title` (not `section`) when calling `summarize_article_section`. Otherwise, a validation error will occur."*

**GEPA self-discovered the MCP server's schema quirks by reflecting on baseline failures and patched them into the prompt.** This is the exact failure mode mcparena is built to surface: MCP servers ship with tool descriptions that under-specify their argument schemas, and an optimizer that observes real trajectories can recover knowledge the base model couldn't infer. The +12.5pp lift is the side-effect; the demonstrable mechanism is the point.

The full vision — public leaderboard, server-author outreach, GitHub Action CI gate — ships once the pilot stabilizes a second-server signal and the schema-discovery story is reproducible.

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
```

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
