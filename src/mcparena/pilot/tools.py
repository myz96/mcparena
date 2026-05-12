"""MCP tool helpers — closure pattern, async/sync bridge, axis ii/iii."""

from __future__ import annotations

import asyncio
import itertools
import random
from typing import Any

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


def discover_tool_specs(params: StdioServerParameters) -> list[dict[str, Any]]:
    """Briefly connect to an MCP server, list its tools, return name/description/schema dicts."""

    async def _list() -> list[dict[str, Any]]:
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                resp = await session.list_tools()
                return [
                    {
                        "name": t.name,
                        "description": t.description or "",
                        "inputSchema": t.inputSchema,
                    }
                    for t in resp.tools
                ]

    return asyncio.run(_list())


def make_tools(params: StdioServerParameters) -> list[Any]:
    """Build `dspy.Tool` objects wrapping each MCP tool via closure.

    MCP SDK is async-only but `dspy.ReAct.forward` is sync — we bridge via
    `asyncio.run` per tool call inside a short-lived stdio session.
    """
    import dspy

    specs = discover_tool_specs(params)
    tools: list[Any] = []
    for spec in specs:
        name = spec["name"]
        desc = spec.get("description", "")
        tools.append(dspy.Tool(_make_caller(params, name), name=name, desc=desc))
    return tools


def _make_caller(params: StdioServerParameters, tool_name: str) -> Any:
    """Build a sync callable invoking `session.call_tool(tool_name, **kwargs)`.

    Captured-by-default-arg pattern avoids late-binding bugs in the calling loop.
    """

    def _call(**kwargs: Any) -> str:
        async def _ainvoke() -> Any:
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    return await session.call_tool(tool_name, arguments=kwargs)

        result = asyncio.run(_ainvoke())
        if not result.content:
            return ""
        parts: list[str] = []
        for content_item in result.content:
            text = getattr(content_item, "text", None)
            if text is not None:
                parts.append(text)
        return "\n".join(parts)

    _call.__name__ = tool_name
    return _call


def permute_tools(
    tools: list[Any],
    max_permutations: int = 4,
    seed: int = 0,
) -> list[list[Any]]:
    """Generate up to N permutations of the tool list for axis (ii).

    Always returns the original order first. Exhaustive for small lists;
    otherwise samples with a fixed seed for reproducibility.
    """
    rng = random.Random(seed)
    original = list(tools)
    all_perms = list(itertools.permutations(original))
    if len(all_perms) <= max_permutations:
        return [list(p) for p in all_perms]
    others = [p for p in all_perms if list(p) != original]
    sampled = rng.sample(others, max_permutations - 1)
    return [original] + [list(p) for p in sampled]


def inject_one_shot(
    tools: list[Any],
    exemplar_calls: dict[str, dict[str, Any]],
) -> list[Any]:
    """Return new `dspy.Tool` list with exemplar usage injected into descriptions.

    `exemplar_calls` maps tool name -> {"input": kwargs, "output": text-result}.
    Tools without an exemplar pass through unchanged.
    """
    import dspy

    new_tools: list[Any] = []
    for t in tools:
        ex = exemplar_calls.get(t.name)
        if ex is None:
            new_tools.append(t)
            continue
        addition = (
            f"\n\nExample usage:\n"
            f"  call: {t.name}({ex.get('input', {})})\n"
            f"  result: {ex.get('output', '')!r}"
        )
        existing_desc = t.desc or ""
        new_tools.append(dspy.Tool(t.func, name=t.name, desc=existing_desc + addition))
    return new_tools
