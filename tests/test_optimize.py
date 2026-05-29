"""Unit tests for mcparena.optimize — hermetic; no MCP / LLM access."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mcparena import optimize


def test_build_examples_requires_user_request() -> None:
    with pytest.raises(ValueError, match="missing required field 'user_request'"):
        optimize._build_examples([{"task_id": "t0"}])


def test_build_examples_assigns_default_task_id_when_missing() -> None:
    examples = optimize._build_examples(
        [{"user_request": "first"}, {"task_id": "custom", "user_request": "second"}]
    )
    assert len(examples) == 2
    assert examples[0].task_id == "task_0"
    assert examples[1].task_id == "custom"
    assert examples[0].user_request == "first"


def test_replicate_expands_examples_n_times() -> None:
    assert optimize._replicate(["a", "b"], n_trials=3) == ["a", "b", "a", "b", "a", "b"]


def test_replicate_zero_returns_empty() -> None:
    assert optimize._replicate(["x"], n_trials=0) == []


def test_format_eval_extracts_scores() -> None:
    fake = SimpleNamespace(
        results=[
            (SimpleNamespace(), SimpleNamespace(), 1.0),
            (SimpleNamespace(), SimpleNamespace(), 0.0),
            (SimpleNamespace(), SimpleNamespace(), 0.5),
        ]
    )
    result = optimize._format_eval(fake, "baseline")
    assert result["condition"] == "baseline"
    assert result["n_trials"] == 3
    assert result["mean_score"] == 0.5
    assert result["per_trial_scores"] == [1.0, 0.0, 0.5]


def test_format_eval_handles_no_results() -> None:
    result = optimize._format_eval(SimpleNamespace(), "gepa")
    assert result["mean_score"] == 0.0
    assert result["n_trials"] == 0


def test_extract_discovered_prompt_returns_none_when_no_predict() -> None:
    assert optimize._extract_discovered_prompt(SimpleNamespace()) is None


def test_extract_discovered_prompt_walks_react_signature_instructions() -> None:
    fake = SimpleNamespace(
        react=SimpleNamespace(signature=SimpleNamespace(instructions="EVOLVED PROMPT"))
    )
    assert optimize._extract_discovered_prompt(fake) == "EVOLVED PROMPT"


def test_extract_discovered_prompt_falls_back_to_predict_attr() -> None:
    fake = SimpleNamespace(
        predict=SimpleNamespace(signature=SimpleNamespace(instructions="ALT PATH"))
    )
    assert optimize._extract_discovered_prompt(fake) == "ALT PATH"


def test_bootstrap_delta_empty_returns_error() -> None:
    assert optimize._bootstrap_delta([], [1.0])["error"] == "no paired samples"
    assert optimize._bootstrap_delta([1.0], [])["error"] == "no paired samples"


def test_bootstrap_delta_clear_lift() -> None:
    # GEPA wins every paired trial — point estimate must be +100pp and CI in (-inf, +inf)
    result = optimize._bootstrap_delta(
        baseline_scores=[0.0, 0.0, 0.0, 0.0], gepa_scores=[1.0, 1.0, 1.0, 1.0]
    )
    assert result["delta_pp"] == pytest.approx(100.0)
    assert result["n_paired"] == 4


def test_bootstrap_delta_pairs_to_minimum_length() -> None:
    result = optimize._bootstrap_delta(
        baseline_scores=[0.0, 0.0, 0.0, 0.0, 0.0], gepa_scores=[1.0, 0.0, 1.0]
    )
    assert result["n_paired"] == 3


def test_optimize_config_to_stdio_params_passes_through(tmp_path: Path) -> None:
    cfg = optimize.OptimizeConfig(
        server_cmd="python",
        server_args=["-m", "wikipedia_mcp"],
        tasks=[{"user_request": "x"}],
        server_env={"FOO": "bar"},
        server_cwd=str(tmp_path),
    )
    params = cfg.to_stdio_params()
    assert params.command == "python"
    assert params.args == ["-m", "wikipedia_mcp"]
    assert params.env == {"FOO": "bar"}
    assert params.cwd == str(tmp_path)


def test_load_config_from_args_reads_json(tmp_path: Path) -> None:
    tasks_path = tmp_path / "tasks.json"
    tasks_path.write_text(json.dumps([{"user_request": "find pi to 5 digits"}]))
    cfg = optimize.load_config_from_args(
        server_cmd="node",
        server_args=["build/index.js"],
        tasks_path=tasks_path,
        n_trials=2,
        output_dir=tmp_path / "out",
        max_full_evals=1,
        server_cwd=None,
    )
    assert cfg.tasks == [{"user_request": "find pi to 5 digits"}]
    assert cfg.n_trials == 2
    assert cfg.output_dir == tmp_path / "out"


def test_load_config_from_args_rejects_non_list_json(tmp_path: Path) -> None:
    tasks_path = tmp_path / "bad.json"
    tasks_path.write_text(json.dumps({"tasks": [{"user_request": "x"}]}))
    with pytest.raises(ValueError, match="must contain a JSON list"):
        optimize.load_config_from_args(
            server_cmd="x",
            server_args=[],
            tasks_path=tasks_path,
            n_trials=1,
            output_dir=tmp_path,
            max_full_evals=1,
            server_cwd=None,
        )


def test_write_summary_markdown_contains_constraints_block(tmp_path: Path) -> None:
    out = tmp_path / "summary.md"
    optimize._write_summary_markdown(
        out,
        baseline={"mean_score": 0.0, "n_trials": 16, "per_trial_scores": [0.0] * 16},
        gepa={"mean_score": 0.125, "n_trials": 16, "per_trial_scores": [0.0] * 14 + [1.0, 1.0]},
        delta={"delta_pp": 12.5, "ci_low_pp": 0.0, "ci_high_pp": 31.25, "n_paired": 16},
        discovered_prompt="### Critical Task Constraints:\n1. Use foo not bar",
        cost_usd=6.2906,
    )
    content = out.read_text()
    assert "+12.50pp" in content
    assert "[+0.00pp, +31.25pp]" in content
    assert "$6.2906" in content
    assert "Critical Task Constraints" in content
    assert "Use foo not bar" in content


def test_write_summary_markdown_omits_prompt_block_when_none(tmp_path: Path) -> None:
    out = tmp_path / "summary.md"
    optimize._write_summary_markdown(
        out,
        baseline={"mean_score": 0.5, "n_trials": 2, "per_trial_scores": [1.0, 0.0]},
        gepa={"mean_score": 0.5, "n_trials": 2, "per_trial_scores": [1.0, 0.0]},
        delta={"delta_pp": 0.0, "ci_low_pp": 0.0, "ci_high_pp": 0.0, "n_paired": 2},
        discovered_prompt=None,
        cost_usd=0.42,
    )
    assert "What GEPA discovered" not in out.read_text()


def test_cli_optimize_requires_server_cmd_and_tasks(capsys: pytest.CaptureFixture[str]) -> None:
    from mcparena.cli import _build_parser

    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["optimize"])
    err = capsys.readouterr().err
    assert "--server-cmd" in err or "required" in err.lower()


def test_cli_optimize_parses_args(tmp_path: Path) -> None:
    from mcparena.cli import _build_parser

    tasks_path = tmp_path / "t.json"
    tasks_path.write_text("[]")
    parser = _build_parser()
    args = parser.parse_args(
        [
            "optimize",
            "--server-cmd",
            "python -m wikipedia_mcp",
            "--tasks",
            str(tasks_path),
            "--n-trials",
            "5",
        ]
    )
    assert args.command == "optimize"
    assert args.server_cmd == "python -m wikipedia_mcp"
    assert args.tasks == tasks_path
    assert args.n_trials == 5
    assert args.max_full_evals == 1
