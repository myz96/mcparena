# mcparena

> The optimization layer over MCP benchmarks. Improve your MCP server's score on [MCP-Bench](https://github.com/Accenture/mcp-bench) via [GEPA](https://github.com/gepa-ai/gepa) + [DSPy](https://github.com/stanfordnlp/dspy).

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![Status](https://img.shields.io/badge/Status-Alpha-orange.svg)](#status)

## Status

**Alpha — pilot phase, public for visibility.** This is a pre-registered, in-progress spike validating whether DSPy optimizers (GEPA, MIPROv2) and tool-list mechanisms (ordering permutations, 1-shot example injection) can meaningfully improve MCP server tool-use success rates on the [MCP-Bench](https://github.com/Accenture/mcp-bench) benchmark.

### Pilot results so far

| Server | Baseline (vanilla `dspy.ReAct`) | GEPA | Δ |
|---|---|---|---|
| Math MCP (easy) | 75% (3/4) | 75% (matched) | 0pp — saturated, no headroom |
| Wikipedia (medium) | 16.7% (1/6) | **33.3% (2/6)** | **+16.7pp** |
| OpenAPI Explorer (hard) | infeasible | — | single tool response = 1.6M tokens, exceeds 256K context |

Wikipedia shows a real point-estimate lift, but n=6 → wide CI ([-33.3pp, +66.7pp]). Phase 2 (n=15) is planned to tighten the confidence interval. Per-condition cost on Qwen3-235b: ~$0.50-4 actual (Sonnet would have been 50-150× higher).

The full vision — public leaderboard, server-author outreach, GitHub Action CI gate — ships only after the pilot decision gate is met.

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
