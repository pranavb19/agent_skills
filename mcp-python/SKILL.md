---
name: mcp-python
description: Build MCP (Model Context Protocol) servers and clients in Python using the official `mcp` SDK and FastMCP. Use this skill whenever the user mentions MCP, Model Context Protocol, MCP server, MCP client, MCP tools, MCP resources, MCP prompts, FastMCP, streamable HTTP transport, STDIO transport, MCP sampling, MCP elicitation, MCP roots, MultiServerMCPClient, langchain-mcp-adapters, or wants to connect an AI agent to external tools/data via MCP. Also use when building agentic systems that need tool integration, even if the user doesn't explicitly say "MCP".
---

# MCP Python Development Skill

Build MCP servers and clients in Python using the `mcp` SDK (which includes FastMCP).

## Quick orientation

- **Package**: `mcp` on PyPI (includes FastMCP). Install: `pip install "mcp[cli]"`
- **For LangChain/LangGraph integration**: `pip install langchain-mcp-adapters`
- **Spec versions**: `2024-11-05`, `2025-03-26`, `2025-06-18`, `2025-11-25`
- **Imports**: `from mcp.server.fastmcp import FastMCP, Context, Image`

## Before you start coding

Read the appropriate reference file(s) from this skill's `references/` directory based on what the user needs:

| User wants to... | Read this file |
|---|---|
| Create an MCP server (tools, resources, prompts) | `references/server.md` |
| Create an MCP client or multi-server agent | `references/client.md` |
| Use Context (logging, progress, sampling, elicitation) | `references/context.md` |
| Understand transports, auth, security, lifecycle | `references/protocol.md` |

**Always read the relevant reference file before writing code.** The references contain exact import paths, API signatures, return type mappings, and gotchas that are critical for correct code.

## Architecture at a glance

```
Host (Claude Desktop / IDE / custom agent)
├── MCP Client 1 ──► MCP Server A (local, STDIO)
├── MCP Client 2 ──► MCP Server B (remote, Streamable HTTP)
└── MCP Client 3 ──► MCP Server C (remote, OAuth 2.1)
```

- **Host**: AI application that creates clients
- **Client**: 1:1 connection to a single server
- **Server**: Exposes tools, resources, prompts

Three primitives: **Tools** (model-controlled functions), **Resources** (app-controlled read-only data), **Prompts** (user-controlled message templates).

Two transports: **STDIO** (local subprocess, single client) and **Streamable HTTP** (network, multi-client, default endpoint `/mcp`).

## Minimal server template (copy-paste starter)

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("MyServer")

@mcp.tool()
def my_tool(param: str) -> str:
    """Describe what this tool does."""
    return f"Result: {param}"

@mcp.resource("data://config")
def get_config() -> str:
    """Server configuration."""
    return '{"key": "value"}'

if __name__ == "__main__":
    mcp.run()  # STDIO by default; use mcp.run(transport="streamable-http") for HTTP
```

## Minimal client template

```python
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import asyncio

async def main():
    params = StdioServerParameters(command="python", args=["server.py"])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool("my_tool", {"param": "hello"})
            print(result.content[0].text)

asyncio.run(main())
```

## Key rules and gotchas

1. **Docstrings are required** on `@mcp.tool()` functions — they become the tool description sent to the LLM. No docstring = the LLM doesn't know what the tool does.
2. **Type hints are required** — FastMCP generates JSON Schema from them. Use `Annotated[str, Field(description="...")]` for rich parameter docs.
3. **Context is injected by type hint** — add `ctx: Context` as a parameter; it's auto-excluded from the tool's input schema.
4. **Return types matter**: `str` → TextContent, `dict` → JSON TextContent, `bytes` → BlobResourceContents, `Image` → ImageContent, `list` → multiple content blocks.
5. **Errors**: raise exceptions in tools → returned to LLM with `isError=True`. Don't return error strings as normal output.
6. **STDIO servers must never print to stdout** — all non-MCP output goes to stderr. Use `logging` with stderr handler or `ctx.info()`.
7. **Async is preferred** for tools that do I/O. Both sync and async handlers are supported.
8. **Tool annotations are hints, not enforcement**: `readOnlyHint`, `destructiveHint`, `idempotentHint`, `openWorldHint`.
9. **MultiServerMCPClient (v0.1.0+)** is NOT an async context manager. Instantiate directly: `client = MultiServerMCPClient({...})`.
10. **InMemorySaver is dev-only** — use PostgresSaver or SqliteSaver in production.

## Development workflow

```bash
mcp dev server.py                    # Launch with MCP Inspector (debugging UI)
mcp dev server.py --with pandas      # With extra dependencies
mcp install server.py                # Register in Claude Desktop
mcp install server.py -e API_KEY=x   # With env vars
mcp run server.py                    # Run directly
```

## When to use which transport

| Scenario | Transport | Why |
|---|---|---|
| Local CLI tool / desktop integration | STDIO | Zero network, microsecond latency |
| Production API, multi-client | Streamable HTTP | Scalable, load-balanced |
| Serverless (Lambda, Cloud Run) | Streamable HTTP + `stateless_http=True` | Stateless per-request |
| Embedding in existing FastAPI/Starlette | Streamable HTTP (mount) | Shares ASGI app |
