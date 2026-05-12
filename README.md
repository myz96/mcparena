# mcparena

> The optimization layer over MCP benchmarks. Improve your MCP server's score on [MCP-Bench](https://github.com/Accenture/mcp-bench) (and friends) via [GEPA](https://github.com/gepa-ai/gepa) + [DSPy](https://github.com/stanfordnlp/dspy).

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![Status](https://img.shields.io/badge/Status-Alpha-orange.svg)](#status)
<!-- CI badge added after first push to GitHub -->

## Status

**Pre-launch pilot in progress.** This repo is the scaffold for a 3-4 day spike validating whether DSPy optimizers (GEPA, MIPROv2) and additional mechanisms (tool ordering, 1-shot example in tool description) can meaningfully improve MCP server tool-use success rates on the [MCP-Bench](https://github.com/Accenture/mcp-bench) benchmark.

The full vision (public leaderboard, server-author outreach, GitHub Action for CI, viral launch) ships only after the pilot decision gate.

## Project ethos

- **Pre-registration discipline.** The pilot metric, decision criteria, and MCP-Bench task IDs are committed to `docs/pilot/pre_registration.md` and locked to git tag `pilot-prereg-v1` *before* any results exist. Verify via `git log`.
- **Build on existing benchmarks.** mcparena uses [MCP-Bench (Accenture, NeurIPS 2025)](https://github.com/Accenture/mcp-bench) as ground truth rather than reinventing measurement. Future versions integrate MCP-Universe, MCPMark, MCPVerse.
- **Multiple mechanisms.** Pilot tests five conditions: baseline / MIPROv2 / GEPA / tool ordering / 1-shot example. Strong soft-fail mitigation — if one mechanism stagnates, others may lift.

## Quickstart

```bash
# Install dependencies
uv sync

# Smoke test the GEPA adapter (~$1, validates it loads on Filesystem MCP)
mcparena pilot --smoke-adapter

# Calibrate cost projections (~$10, 1 server, 5 conditions)
mcparena pilot --smoke-budget

# Full pilot (~$150 expected, $350 hard cap)
mcparena pilot
```

## Architecture

| Layer | Tool |
|---|---|
| Program / judge LM | Claude Sonnet 4.6 |
| GEPA reflection LM | Claude Opus 4.7 |
| Optimizers | DSPy MIPROv2 + GEPA (ICLR 2026 oral, native MCP adapter) |
| Task source | MCP-Bench (Accenture) |
| Statistical aggregation | `scipy.stats.bootstrap` (paired 95% CI, n_resamples=1000) |
| MCP transport | stdio (pilot); HTTP/SSE (Phase 1) |

## Docs

- `docs/pilot/pre_registration.md` — locked pilot metric + decision criteria (forthcoming)
- `docs/plans/` — cleaned design + plan docs (forthcoming; raw drafts live in `docs/_drafts/` gitignored)

## License

[MIT](LICENSE)
