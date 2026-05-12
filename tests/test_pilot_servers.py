"""PILOT_SERVERS shape + stratification coverage."""

from mcp.client.stdio import StdioServerParameters

from mcparena.pilot import PILOT_SERVERS, ServerSpec


def test_three_servers_defined() -> None:
    assert len(PILOT_SERVERS) == 3, "Pilot stratifies across 3 servers (easy/medium/hard)"
    for spec in PILOT_SERVERS:
        assert isinstance(spec, ServerSpec)


def test_stratification_covers_all_tiers() -> None:
    """Plan v5.1: one server per difficulty_tier in {easy, medium, hard}."""
    tiers = {spec.difficulty_tier for spec in PILOT_SERVERS}
    assert tiers == {
        "easy",
        "medium",
        "hard",
    }, f"Expected stratified coverage of easy/medium/hard; got {tiers}"


def test_each_spec_produces_stdio_params() -> None:
    for spec in PILOT_SERVERS:
        params = spec.to_stdio_params()
        assert isinstance(params, StdioServerParameters)
        assert params.command == spec.command
        assert list(params.args) == spec.args
