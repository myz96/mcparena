"""MCP tool helpers — persistent session, async/sync bridge, axis ii/iii.

The MCP Python SDK is async-only but `dspy.ReAct.forward` is sync. The
previous per-call closure spawned a fresh stdio subprocess on every tool
invocation — at pilot scale (~thousands of calls) this exhausts resource
trackers (~1.7k subprocess spawns silently hung at smoke-budget on 2026-05-13).

`PersistentMCPSession` keeps the ClientSession alive for the entire duration
of a `run_*` function. A dedicated thread runs an asyncio event loop; sync
tool callers submit coroutines via `asyncio.run_coroutine_threadsafe`. Calls
are serialized with a lock — the MCP SDK's `ClientSession.call_tool` is not
documented as concurrency-safe.
"""

from __future__ import annotations

import asyncio
import itertools
import random
import threading
from contextlib import contextmanager
from typing import Any

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


def discover_tool_specs(params: StdioServerParameters) -> list[dict[str, Any]]:
    """Briefly connect to an MCP server, list its tools, return name/description/schema dicts.

    Used by `run_smoke_adapter` (R9 gate) to verify connectivity without
    paying for a persistent session. Production callers use
    `PersistentMCPSession.tool_specs` instead.
    """

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


class PersistentMCPSession:
    """Keeps an MCP `ClientSession` alive across thread boundaries.

    Lifecycle:
      with PersistentMCPSession(params) as session:
          tools = make_tools(session)        # dspy.Tool list
          # ... DSPy may call session.call_tool concurrently from N threads
      # session closed; subprocess reaped on exit

    The event loop runs in a daemon thread; sync `call_tool` submits
    coroutines via `run_coroutine_threadsafe` and waits up to `timeout` for
    a response.
    """

    def __init__(self, params: StdioServerParameters, call_timeout: float = 60.0) -> None:
        self.params = params
        self.call_timeout = call_timeout
        self.loop: asyncio.AbstractEventLoop | None = None
        self.thread: threading.Thread | None = None
        self.session: ClientSession | None = None
        self._tools: list[dict[str, Any]] = []
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._error: BaseException | None = None
        self._call_lock = threading.Lock()

    @property
    def tool_specs(self) -> list[dict[str, Any]]:
        return list(self._tools)

    def __enter__(self) -> PersistentMCPSession:
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        if not self._ready.wait(timeout=30):
            raise RuntimeError("MCP server failed to initialize within 30s")
        if self._error is not None:
            raise self._error
        return self

    def __exit__(self, *_args: Any) -> None:
        self._stop.set()
        if self.thread is not None:
            self.thread.join(timeout=15)

    def call_tool(self, tool_name: str, **kwargs: Any) -> str:
        """Synchronously call a tool. Serialized via internal lock for SDK safety."""
        if self.session is None or self.loop is None:
            raise RuntimeError("PersistentMCPSession is not open")
        with self._call_lock:
            future = asyncio.run_coroutine_threadsafe(
                self.session.call_tool(tool_name, arguments=kwargs),
                self.loop,
            )
            result = future.result(timeout=self.call_timeout)
        if not result.content:
            return ""
        parts: list[str] = []
        for content_item in result.content:
            text = getattr(content_item, "text", None)
            if text is not None:
                parts.append(text)
        return "\n".join(parts)

    def _run_loop(self) -> None:
        try:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            self.loop.run_until_complete(self._open_and_serve())
        except BaseException as exc:
            self._error = exc
            self._ready.set()
        finally:
            if self.loop is not None and not self.loop.is_closed():
                self.loop.close()

    async def _open_and_serve(self) -> None:
        async with stdio_client(self.params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                self.session = session
                resp = await session.list_tools()
                self._tools = [
                    {
                        "name": t.name,
                        "description": t.description or "",
                        "inputSchema": t.inputSchema,
                    }
                    for t in resp.tools
                ]
                self._ready.set()
                while not self._stop.is_set():
                    await asyncio.sleep(0.1)


@contextmanager
def persistent_session(params: StdioServerParameters) -> Any:
    """Convenience context manager around `PersistentMCPSession`."""
    session = PersistentMCPSession(params)
    with session as opened:
        yield opened


def make_tools(session: PersistentMCPSession) -> list[Any]:
    """Build `dspy.Tool` objects wrapping each MCP tool via the persistent session."""
    import dspy

    return [
        dspy.Tool(_make_caller(session, spec["name"]), name=spec["name"], desc=spec["description"])
        for spec in session.tool_specs
    ]


def _make_caller(session: PersistentMCPSession, tool_name: str) -> Any:
    """Build a sync callable that delegates to `session.call_tool(tool_name, **kwargs)`.

    The closure captures `session` and `tool_name` from the enclosing function
    scope — fresh values per `_make_caller` call (no late-binding risk).
    Default-arg capture was tried but `dspy.Tool` runs pydantic schema
    introspection on the signature and chokes on the `PersistentMCPSession`
    type annotation (PydanticSchemaGenerationError).
    """

    def _call(**kwargs: Any) -> str:
        return session.call_tool(tool_name, **kwargs)

    _call.__name__ = tool_name
    return _call


def permute_tools(
    tools: list[Any],
    max_permutations: int = 4,
    seed: int = 0,
) -> list[list[Any]]:
    """Generate up to N permutations of the tool list for axis (ii).

    Always returns the original order first, then random shuffles to reach N
    distinct permutations. Pure rejection sampling: do NOT materialize the
    full permutation set — math_mcp ships 13 tools (13! = 6.2 billion).
    """
    import math

    rng = random.Random(seed)
    original = list(tools)
    n = len(original)

    if math.factorial(n) <= max_permutations:
        return [list(p) for p in itertools.permutations(original)]

    # For large n, dspy.Tool isn't hashable so we can't dedupe a seen-set.
    # Collision probability for k=4 random shuffles of n>=5 items is ~k²/2n!,
    # which is negligible (e.g. 4²/2·13! = 1.3e-9). Skip dedupe.
    perms: list[list[Any]] = [original]
    for _ in range(max_permutations - 1):
        shuffled = original[:]
        rng.shuffle(shuffled)
        perms.append(shuffled)
    return perms


def inject_one_shot(
    tools: list[Any],
    exemplar_calls: dict[str, dict[str, Any]],
) -> list[Any]:
    """Return new `dspy.Tool` list with exemplar usage injected into descriptions."""
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
        new_tools.append(dspy.Tool(t.func, name=t.name, desc=(t.desc or "") + addition))
    return new_tools
