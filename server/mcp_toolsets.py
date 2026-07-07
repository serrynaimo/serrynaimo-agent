"""Lazy-loaded MCP toolsets for the voice agent.

MCP servers can expose dozens of tools; registering them all upfront would
permanently bloat the LLM context. Instead, only one meta-tool (load_toolset)
is always present. When the model decides a toolset is relevant, the handler:

1. spawns the MCP server (stdio) and keeps the session open,
2. converts its MCP tools to Pipecat FunctionSchemas with proxy handlers,
3. registers the handlers on the LLM service and adds the schemas to the live
   LLMContext,

so the tools exist in context only from that moment on. Add future servers by
extending TOOLSETS.
"""

import asyncio
import os
from datetime import datetime

from loguru import logger
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema

HOME = os.path.expanduser("~")

# Registry of available MCP servers. "when" feeds the load_toolset description
# so the model knows what each toolset is for WITHOUT loading it.
TOOLSETS: dict[str, dict] = {
    "apple-mail": {
        "command": "uvx",
        "args": ["mcp-apple-mail"],
        "when": "reading, searching, organizing, or sending email via Apple Mail",
    },
    "calendar": {
        "command": "uv",
        "args": [
            "--directory", os.path.join(HOME, "Sites", "apple-calendar-mcp"),
            "run", "apple-calendar-mcp",
        ],
        "when": "viewing, creating, or changing calendar events and checking availability",
    },
}

MAX_TOOL_RESULT_CHARS = 6000  # protect the context window from huge outputs

# Each MCP server is owned by a dedicated, detached asyncio task: the stdio
# client's internal reader/writer tasks live inside whichever task entered the
# context, so owning them in a per-server manager task keeps connections alive
# across client sessions (the first version parented them to a tool-call task,
# and disconnecting the client killed the connection).
_managers: dict[str, dict] = {}            # server key -> manager entry
_loaded_schemas: dict[str, list] = {}      # server key -> [FunctionSchema]


async def _server_task(key: str, entry: dict):
    """Own the MCP server connection for its whole lifetime."""
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    spec = TOOLSETS[key]
    try:
        params = StdioServerParameters(
            command=spec["command"], args=spec["args"], env=os.environ.copy()
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                entry["session"] = session
                entry["ready"].set()
                await entry["stop"].wait()
    except Exception as exc:  # noqa: BLE001
        entry["error"] = str(exc)
    finally:
        entry["session"] = None
        entry["ready"].set()
        logger.info(f"MCP: {key} connection closed")


async def _get_session(key: str):
    """Return a live session for the server, (re)connecting as needed."""
    entry = _managers.get(key)
    if entry is not None and entry["session"] is not None:
        try:
            await asyncio.wait_for(entry["session"].send_ping(), timeout=5)
            return entry["session"]
        except Exception:  # noqa: BLE001 — dead server/pipe: reconnect
            logger.warning(f"MCP: {key} session unresponsive, reconnecting")
            entry["stop"].set()
            _managers.pop(key, None)
    spec = TOOLSETS[key]
    logger.info(f"MCP: starting {key} ({spec['command']} {' '.join(spec['args'])})")
    entry = {
        "session": None,
        "ready": asyncio.Event(),
        "stop": asyncio.Event(),
        "error": None,
    }
    # create_task (not awaiting in-place) detaches the connection's lifetime
    # from the calling tool-handler task.
    entry["task"] = asyncio.create_task(_server_task(key, entry), name=f"mcp-{key}")
    _managers[key] = entry
    await asyncio.wait_for(entry["ready"].wait(), timeout=120)
    if entry["session"] is None:
        _managers.pop(key, None)
        raise RuntimeError(entry["error"] or f"MCP server {key} failed to start")
    return entry["session"]


async def call_mcp_tool(key: str, tool_name: str, arguments: dict) -> str:
    """Call one MCP tool directly (server-side, outside the LLM tool flow).

    Spawns/revives the server session on demand — usable before the eager
    toolset registration has run. Returns the tool's text output.
    """
    session = await _get_session(key)
    res = await session.call_tool(tool_name, arguments)
    texts = [c.text for c in res.content if hasattr(c, "text") and c.text]
    out = "\n".join(texts).strip()
    if getattr(res, "isError", False):
        raise RuntimeError(out or f"{tool_name} returned an error")
    return out


def _clean_arguments(arguments: dict) -> dict:
    """Drop no-value arguments the model emits for 'no filter'.

    Small models write optional params as the STRINGS "null"/"none"/"" —
    e.g. account="null" — which downstream servers treat as literal filter
    values (an Apple Mail account named "null") and silently match nothing.
    """
    return {
        k: v
        for k, v in arguments.items()
        if not (isinstance(v, str) and v.strip().lower() in ("", "null", "none")) and v is not None
    }


def _proxy(key: str, tool_name: str):
    """Build a Pipecat tool handler that forwards to the MCP session."""

    async def handler(params):
        arguments = _clean_arguments(dict(params.arguments))
        if arguments != dict(params.arguments):
            logger.info(f"MCP {key}: dropped no-value arguments from {tool_name}")
        logger.info(f"MCP {key}: {tool_name}({arguments})")
        try:
            session = await _get_session(key)  # revives dead connections
            res = await session.call_tool(tool_name, arguments)
        except Exception as exc:  # noqa: BLE001
            await params.result_callback({"error": f"{tool_name} failed: {exc}"})
            return
        texts = [c.text for c in res.content if hasattr(c, "text") and c.text]
        out = "\n".join(texts).strip()
        if len(out) > MAX_TOOL_RESULT_CHARS:
            out = out[:MAX_TOOL_RESULT_CHARS] + "\n[output truncated]"
        result = {"result": out or "(empty result)"}
        if getattr(res, "isError", False):
            result = {"error": out or f"{tool_name} returned an error"}
        # Fresh time on every result, like the native tools: calendar and
        # mail answers are exactly where "how long until X" math happens,
        # and the system prompt only has the session-start time.
        result["current_time"] = datetime.now().astimezone().strftime("%H:%M")
        await params.result_callback(result)

    return handler


async def ensure_toolset_schemas(key: str, base_schemas: list) -> list:
    """Fetch (and cache) a toolset's FunctionSchemas without a live session.

    Shared by load_toolset_impl and the /api/chat text endpoint. Names that
    collide with already-known tools get a toolset prefix, exactly as before.
    """
    if key not in TOOLSETS:
        raise ValueError(f"unknown toolset {key!r}")
    if key not in _loaded_schemas:
        session = await _get_session(key)
        listing = await session.list_tools()
        existing = {s.name for schemas in _loaded_schemas.values() for s in schemas}
        existing |= {s.name for s in base_schemas}
        schemas = []
        for tool in listing.tools:
            name = (
                tool.name
                if tool.name not in existing
                else f"{key.replace('-', '_')}_{tool.name}"
            )
            input_schema = tool.inputSchema or {}
            # First paragraph of the docstring only: Args/Returns sections
            # duplicate the parameter schema and just burn prompt tokens.
            raw_desc = (tool.description or name).strip()
            summary = " ".join(raw_desc.split("\n\n")[0].split())[:300]
            schemas.append(
                FunctionSchema(
                    name=name,
                    description=summary,
                    properties=input_schema.get("properties", {}),
                    required=input_schema.get("required", []),
                )
            )
            # remember the original MCP tool name for the proxy
            schemas[-1]._mcp_tool_name = tool.name  # type: ignore[attr-defined]
        _loaded_schemas[key] = schemas
    return _loaded_schemas[key]


def proxy_handler(key: str, tool_name: str):
    """Public accessor for the MCP proxy handler (used by /api/chat)."""
    return _proxy(key, tool_name)


async def load_toolset_impl(key: str, llm, context, base_schemas: list) -> dict:
    """Connect an MCP server and inject its tools into the live context.

    Safe to call again in a new client session: handlers are (re)registered on
    the CURRENT llm service and schemas (re)injected into the CURRENT context
    every time — tool listings are cached, connections are health-checked and
    revived if the server died.
    """
    if key not in TOOLSETS:
        return {"error": f"unknown toolset {key!r}", "available": sorted(TOOLSETS)}

    first_load = key not in _loaded_schemas
    if first_load:
        try:
            await ensure_toolset_schemas(key, base_schemas)
        except Exception as exc:  # noqa: BLE001
            return {"error": f"could not start toolset {key}: {exc}"}
    # (Re)register handlers on this session's llm and rebuild this context's
    # tool list — required on every call, because reconnects create fresh
    # llm/context objects that know nothing of earlier loads.
    schemas = _loaded_schemas[key]
    for schema in schemas:
        llm.register_function(
            schema.name,
            _proxy(key, getattr(schema, "_mcp_tool_name", schema.name)),
            # Generic interruptions never cancel tool work; only the enrolled
            # speaker does, via the verified-speech hook in bot.py.
            cancel_on_interruption=False,
        )
    all_tools = list(base_schemas)
    for loaded in _loaded_schemas.values():
        all_tools.extend(loaded)
    context.set_tools(ToolsSchema(standard_tools=all_tools))

    logger.info(
        f"MCP: {'loaded' if first_load else 're-attached'} {key} with {len(schemas)} tools"
    )
    # Keep the result tiny: the tool definitions live in the tools array the
    # model already sees, and this result message stays in context forever.
    return {
        "status": "loaded" if first_load else "reloaded",
        "toolset": key,
        "tools_registered": len(schemas),
        "note": "The new tools are available now — call them directly.",
    }


def mcp_tool_names() -> set[str]:
    """Names of all currently-registered MCP tools (grows as toolsets load)."""
    return {s.name for schemas in _loaded_schemas.values() for s in schemas}


def toolset_catalog() -> str:
    """One line per toolset for the load_toolset description."""
    return "; ".join(f"'{k}' for {v['when']}" for k, v in TOOLSETS.items())
