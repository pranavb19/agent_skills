# MCP Context Reference (Python / FastMCP)

The `Context` object gives tool/resource/prompt handlers access to MCP session capabilities. Inject via type hint — automatically excluded from tool input schema.

```python
from mcp.server.fastmcp import FastMCP, Context
```

## Injecting Context

```python
@mcp.tool()
async def my_tool(param: str, ctx: Context) -> str:
    """Tool with context access."""
    # ctx is NOT part of the tool's input schema
    await ctx.info(f"Processing: {param}")
    return "done"
```

## Logging

Send structured log messages to the connected client. Levels: debug, info, warning, error.

```python
@mcp.tool()
async def process(data: str, ctx: Context) -> str:
    """Process data with logging."""
    await ctx.debug(f"Raw input: {data}")
    await ctx.info("Processing started")
    await ctx.warning("Data quality is low")
    await ctx.error("Failed to parse section 3")
    return "done"
```

Logs are sent as `notifications/message` to the client. Also logged to Python logger `fastmcp.server.context.to_client`. Clients can set minimum log level via `logging/setLevel`.

## Progress reporting

Report progress for long-running operations. Silently no-ops if client didn't send a progress token — safe to call unconditionally.

```python
@mcp.tool()
async def batch_process(items: list[str], ctx: Context) -> str:
    """Process items with progress bar."""
    for i, item in enumerate(items):
        await ctx.report_progress(
            progress=i + 1,
            total=len(items),
            message=f"Processing {item}",
        )
        await do_work(item)
    return f"Processed {len(items)} items"
```

## Reading resources from within tools

```python
@mcp.tool()
async def analyze(file_uri: str, ctx: Context) -> str:
    """Analyze a resource."""
    content = await ctx.read_resource(file_uri)
    text = content[0].content  # First content block
    return f"Length: {len(text)}"
```

## Sampling — request LLM completions from the client

Servers can ask the client's host LLM to generate text. The server never needs API keys — the client controls model selection and permissions.

**Flow**: server sends `sampling/createMessage` → client reviews/modifies → client runs LLM → client reviews output → returns result.

### Using ctx.sample() (FastMCP shortcut)

```python
@mcp.tool()
async def smart_summarize(text: str, ctx: Context) -> str:
    """Summarize using the client's LLM."""
    summary = await ctx.sample(
        f"Summarize in one sentence: {text[:1000]}",
        temperature=0.3,
    )
    return summary.text
```

### Using ctx.session.create_message() (full control)

```python
from mcp.types import SamplingMessage, TextContent

@mcp.tool()
async def translate(text: str, target: str, ctx: Context) -> str:
    """Translate text using the client's LLM."""
    result = await ctx.session.create_message(
        messages=[
            SamplingMessage(
                role="user",
                content=TextContent(
                    type="text",
                    text=f"Translate to {target}: {text}"
                ),
            )
        ],
        max_tokens=500,
        model_preferences={
            "hints": [{"name": "claude-3-sonnet"}],  # Advisory preference
            "intelligencePriority": 0.8,  # 0-1 normalized
            "speedPriority": 0.5,
            "costPriority": 0.3,
        },
    )
    return result.content.text
```

### Model preferences (advisory — client decides)

- `hints`: list of `{"name": "substring"}` for model name matching
- `costPriority`: 0-1, how much to optimize for cost
- `speedPriority`: 0-1, how much to optimize for speed
- `intelligencePriority`: 0-1, how much to optimize for capability

## Elicitation — request structured user input

Servers can ask users for structured input through the client. Added in spec 2025-06-18.

```python
@mcp.tool()
async def onboard(ctx: Context) -> str:
    """Collect user details interactively."""
    result = await ctx.elicit(
        message="Please provide your information",
        response_type={
            "name": {"type": "string", "description": "Full name"},
            "email": {"type": "string", "format": "email"},
            "role": {
                "type": "string",
                "enum": ["developer", "designer", "manager"],
            },
            "notify": {"type": "boolean", "description": "Email notifications?"},
        },
    )

    if result.action == "accept":
        return f"Welcome, {result.data['name']}!"
    elif result.action == "decline":
        return "User declined."
    else:  # "cancel"
        return "User cancelled."
```

### Elicitation constraints

- Schema must be a **flat object** — no nested objects or arrays
- Allowed property types: `string`, `number`, `integer`, `boolean`, `enum`
- Three response actions: `"accept"`, `"decline"`, `"cancel"`
- Don't request sensitive info (passwords, SSNs)
- Clients should show which server is requesting and rate-limit requests

## Context API summary

| Method / Property | Purpose |
|---|---|
| `await ctx.debug(msg)` | DEBUG log to client |
| `await ctx.info(msg)` | INFO log to client |
| `await ctx.warning(msg)` | WARNING log to client |
| `await ctx.error(msg)` | ERROR log to client |
| `await ctx.report_progress(progress, total, message)` | Progress notification |
| `await ctx.read_resource(uri)` | Read resource by URI |
| `await ctx.sample(prompt, temperature)` | Quick LLM sampling |
| `await ctx.session.create_message(...)` | Full sampling with preferences |
| `await ctx.elicit(message, response_type)` | Request user input |
| `ctx.request_id` | Current request ID |
| `ctx.client_id` | Client identifier |
| `ctx.session` | Underlying `ServerSession` |
| `ctx.request_context` | Request metadata + lifespan context |
| `ctx.request_context.lifespan_context` | Access lifespan-managed resources |
