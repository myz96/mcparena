"""Verifiable MCP evaluation — the server is its own oracle.

Sample a tool call → execute it against the MCP server → capture the real
result as ground truth → template a task → verify the agent's answer
programmatically. No LLM judge, so no format-gating or temperature noise.

Currently specialized to the deterministic unit-converter server; the
generalization to arbitrary deterministic MCP servers is a later phase.
"""

from mcparena.verifiable.tasks import generate_tasks, uc_stdio_params, verify

__all__ = ["generate_tasks", "uc_stdio_params", "verify"]
