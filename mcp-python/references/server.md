# MCP Server Reference (Python / FastMCP)

## Installation

```bash
pip install "mcp[cli]"        # Core + CLI tools
pip install "mcp[cli,rich]"   # + rich logging
```

## Server initialization

```python
from mcp.server.fastmcp import FastMCP

# Basic
mcp = FastMCP("ServerName")

# With HTTP config
mcp = FastMCP("ServerName", host="0.0.0.0", port=8000)

# Stateless (serverless/Lambda)
mcp = FastMCP("ServerName", stateless_http=True)

# With JSON responses instead of SSE (simpler debugging)
mcp = FastMCP("ServerName", json_response=True)
```

## Tools — `@mcp.tool()`

Tools are model-controlled executable functions. FastMCP auto-generates JSON Schema from type hints and uses docstrings as descriptions.

```python
from mcp.server.fastmcp import FastMCP, Context, Image
from typing import Annotated, Literal
from pydantic import Field

mcp = FastMCP("ToolServer")

# Basic tool — name and schema auto-derived from function
@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b

# Rich parameter metadata with Annotated + Field
@mcp.tool()
def resize_image(
    url: Annotated[str, Field(description="Image URL to resize")],
    width: Annotated[int, Field(ge=1, le=4096, description="Width in pixels")] = 800,
    format: Literal["jpeg", "png", "webp"] = "jpeg",
) -> Image:
    """Resize an image to specified dimensions."""
    raw = do_resize(url, width, format)
    return Image(data=raw, format=format)

# Async tool (preferred for I/O)
@mcp.tool()
async def fetch_page(url: str) -> str:
    """Fetch webpage content."""
    async with aiohttp.ClientSession() as s:
        async with s.get(url) as r:
            return await r.text()

# Custom name + description + behavioral annotations
@mcp.tool(
    name="delete_record",
    description="Permanently delete a database record",
    annotations={
        "destructiveHint": True,
        "idempotentHint": False,
        "readOnlyHint": False,
        "openWorldHint": False,
    },
)
def delete_record(record_id: str) -> dict:
    return {"deleted": record_id}

# Tool with Context for logging/progress (ctx excluded from schema)
@mcp.tool()
async def process_data(file_uri: str, ctx: Context) -> str:
    """Process a data file with progress."""
    await ctx.info("Starting processing")
    data = await ctx.read_resource(file_uri)
    # ... processing ...
    return "Done"
```

### Return type mapping

| Python return type | MCP content type |
|---|---|
| `str` | `TextContent` (plain text) |
| `dict`, Pydantic model | `TextContent` (JSON serialized) |
| `bytes` | `BlobResourceContents` (base64) |
| `Image(data=..., format=...)` | `ImageContent` |
| `list` | Multiple content blocks |
| Raise exception | `CallToolResult` with `isError=True` |

### Tool annotations (advisory hints to clients)

| Annotation | Default | Meaning |
|---|---|---|
| `readOnlyHint` | `false` | Tool only reads, never modifies |
| `destructiveHint` | `true` | Changes are destructive/irreversible |
| `idempotentHint` | `false` | Repeated identical calls are safe |
| `openWorldHint` | `true` | Interacts with external entities |

## Resources — `@mcp.resource()`

Resources are application-controlled, read-only data sources with URI addressing.

```python
import json
from pathlib import Path

# Static resource (fixed URI)
@mcp.resource("config://app/settings")
def get_settings() -> str:
    """Application configuration."""
    return json.dumps({"debug": False, "version": "2.0"})

# Resource template (dynamic URI with RFC 6570 params)
@mcp.resource("users://{user_id}/profile")
def get_profile(user_id: str) -> str:
    """User profile by ID."""
    return json.dumps(load_user(user_id))

# Binary resource with MIME type
@mcp.resource("assets://logo", mime_type="image/png")
def get_logo() -> bytes:
    return Path("logo.png").read_bytes()

# Multi-parameter template
@mcp.resource("db://{database}/{table}/schema")
def table_schema(database: str, table: str) -> str:
    return get_schema(database, table)
```

- Static resources → `resources/list`
- Templates → `resourceTemplates/list`
- Clients read via `resources/read` with concrete URI

## Prompts — `@mcp.prompt()`

Prompts are user-controlled reusable message templates. All arguments are strings over the wire.

```python
from mcp.server.fastmcp.prompts.base import UserMessage, AssistantMessage

# Simple prompt (returns single user message)
@mcp.prompt()
def explain(topic: str) -> str:
    """Ask for a topic explanation."""
    return f"Explain '{topic}' in simple terms."

# Multi-turn prompt
@mcp.prompt()
def debug_error(error: str, language: str = "python") -> list:
    """Start a debugging session."""
    return [
        UserMessage(f"I got this {language} error:\n```\n{error}\n```"),
        AssistantMessage("I'll help debug. Share the full traceback?"),
        UserMessage("Here's the traceback: ..."),
    ]
```

## Running the server

```python
if __name__ == "__main__":
    mcp.run()                              # STDIO (default)
    mcp.run(transport="streamable-http")   # HTTP on configured host:port
```

## Mounting in existing ASGI app (Starlette/FastAPI)

```python
import contextlib
from starlette.applications import Starlette
from starlette.routing import Mount
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Embedded", json_response=True)

@mcp.tool()
def hello() -> str:
    """Say hello."""
    return "Hello from MCP!"

@contextlib.asynccontextmanager
async def lifespan(app):
    async with mcp.session_manager.run():
        yield

app = Starlette(
    routes=[Mount("/", app=mcp.streamable_http_app())],
    lifespan=lifespan,
)
# Run: uvicorn module:app --host 0.0.0.0 --port 8000
```

## Lifespan for shared resources (DB connections, etc.)

```python
from contextlib import asynccontextmanager
from dataclasses import dataclass

@dataclass
class AppContext:
    db: Database

@asynccontextmanager
async def app_lifespan(server: FastMCP):
    db = await Database.connect()
    try:
        yield AppContext(db=db)
    finally:
        await db.disconnect()

mcp = FastMCP("DBServer", lifespan=app_lifespan)

@mcp.tool()
async def query(sql: str, ctx: Context) -> str:
    """Run a SQL query."""
    db = ctx.request_context.lifespan_context["db"]
    return str(await db.execute(sql))
```

## Low-level server API (full protocol control)

Use `mcp.server.lowlevel.Server` when you need direct control over protocol handling.

```python
from mcp.server.lowlevel import Server
import mcp.server.stdio
import mcp.types as types

server = Server("low-level-example")

@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [types.Tool(
        name="greet",
        description="Greet someone",
        inputSchema={
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
    )]

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    if name == "greet":
        return [types.TextContent(type="text", text=f"Hello, {arguments['name']}!")]

async def main():
    async with mcp.server.stdio.stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```
