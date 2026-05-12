"""Smoke test: every pilot module imports cleanly."""


def test_top_level_import() -> None:
    import mcparena

    assert mcparena.__version__ == "0.0.1"


def test_pilot_public_api() -> None:
    from mcparena.pilot import (
        PILOT_SERVERS,
        TASKS_BY_SERVER,
        Assess,
        Condition,
        ServerSpec,
        get_lm,
        judge_metric_evaluate,
        judge_metric_gepa,
    )

    # All callable / non-None
    assert callable(get_lm)
    assert callable(judge_metric_evaluate)
    assert callable(judge_metric_gepa)
    assert isinstance(PILOT_SERVERS, list)
    assert isinstance(TASKS_BY_SERVER, dict)
    assert Assess is not None
    assert ServerSpec is not None
    # Condition is a Literal type, just verify it imports
    assert Condition is not None


def test_cli_module_import() -> None:
    from mcparena import cli

    parser = cli._build_parser()
    # Smoke — parser builds without error and recognizes 'pilot' subcommand
    args = parser.parse_args(["pilot", "--smoke-adapter"])
    assert args.command == "pilot"
    assert args.smoke_adapter is True
