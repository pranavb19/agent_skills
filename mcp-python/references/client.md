# MCP Client Reference (Python)

## STDIO Client

Connect to a local MCP server running as a subprocess.

```python
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import asyncio

server_params = StdioServerParameters(
    command="python",
    args=["server.py"],
    env=None,  # Optional: dict of env vars for subprocess
)

async def main():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # List available tools
            tools = await session.list_tools()

            # Call a tool
            result = await session.call_tool("tool_name", {"param": "value"})
            print(result.content[0].text)

            # List and read resources
            resources = await session.list_resources()
            data = await session.read_resource("config://app/settings")

            # List and get prompts
            prompts = await session.list_prompts()
            prompt = await session.get_prompt("explain", {"topic": "MCP"})

asyncio.run(main())
```

## Streamable HTTP Client

Connect to a remote MCP server over HTTP.

```python
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
import asyncio

async def main():
    async with streamablehttp_client("http://localhost:8000/mcp") as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool("search", {"query": "hello"})
            print(result.content[0].text)

asyncio.run(main())
```

## Paginating list results

All list operations support cursor-based pagination.

```python
async def get_all_tools(session: ClientSession):
    all_tools, cursor = [], None
    while True:
        result = await session.list_tools(cursor=cursor)
        all_tools.extend(result.tools)
        if not result.nextCursor:
            break
        cursor = result.nextCursor
    return all_tools
```

## MultiServerMCPClient (LangChain/LangGraph)

Connect to multiple MCP servers and use their tools in a LangGraph agent.

**Install**: `pip install langchain-mcp-adapters langgraph langchain-anthropic`

```python
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_anthropic import ChatAnthropic
import asyncio

async def main():
    # IMPORTANT: As of v0.1.0+, NOT an async context manager.
    # Instantiate directly.
    client = MultiServerMCPClient({
        "math": {
            "transport": "stdio",
            "command": "python",
            "args": ["math_server.py"],
        },
        "weather": {
            "transport": "http",          # Streamable HTTP
            "url": "http://localhost:8000/mcp",
            "headers": {"Authorization": "Bearer token123"},
        },
    })

    # Get LangChain-compatible tools from all connected servers
    tools = await client.get_tools()

    # Create a ReAct agent
    agent = create_react_agent(
        ChatAnthropic(model="claude-sonnet-4-20250514"),
        tools,
    )

    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "What's 3+5?"}]}
    )
    print(result["messages"][-1].content)

asyncio.run(main())
```

### Transport config options

**STDIO transport**:
```python
{
    "transport": "stdio",
    "command": "python",       # or "node", "npx", etc.
    "args": ["server.py"],
    "env": {"API_KEY": "..."},  # optional
}
```

**HTTP transport**:
```python
{
    "transport": "http",
    "url": "http://host:port/mcp",
    "headers": {"Authorization": "Bearer ..."},  # optional
}
```

**SSE transport** (deprecated, use HTTP):
```python
{
    "transport": "sse",
    "url": "http://host:port/sse",
}
```

### Tool name prefix

Use `tool_name_prefix=True` when multiple servers might define same-named tools:
```python
client = MultiServerMCPClient({...})
tools = await client.get_tools(tool_name_prefix=True)
# Tools named like "math__add", "weather__forecast"
```

## InMemorySaver for conversation state

`InMemorySaver` stores LangGraph checkpoints in memory. Each `thread_id` is an isolated conversation.

**Dev/test only** — data lost on restart. Production: use `PostgresSaver` or `SqliteSaver`.

```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent

checkpointer = InMemorySaver()

agent = create_react_agent(
    ChatAnthropic(model="claude-sonnet-4-20250514"),
    tools,
    checkpointer=checkpointer,
)

# Same thread_id = continuous conversation
config = {"configurable": {"thread_id": "session-1"}}

r1 = await agent.ainvoke(
    {"messages": [{"role": "user", "content": "My name is Alice"}]},
    config,
)
r2 = await agent.ainvoke(
    {"messages": [{"role": "user", "content": "What's my name?"}]},
    config,  # Remembers: "Alice"
)

# Different thread = different conversation
config_b = {"configurable": {"thread_id": "session-2"}}
```

### Production checkpointers

```python
# PostgreSQL
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
checkpointer = AsyncPostgresSaver.from_conn_string("postgresql://...")

# SQLite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
checkpointer = AsyncSqliteSaver.from_conn_string("sqlite:///checkpoints.db")
```

## Full agent example with memory + multi-server

```python
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_anthropic import ChatAnthropic
import asyncio

async def main():
    client = MultiServerMCPClient({
        "filesystem": {
            "transport": "stdio",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
        },
        "api": {
            "transport": "http",
            "url": "http://localhost:8000/mcp",
        },
    })

    tools = await client.get_tools()
    checkpointer = InMemorySaver()

    agent = create_react_agent(
        ChatAnthropic(model="claude-sonnet-4-20250514"),
        tools,
        checkpointer=checkpointer,
    )

    config = {"configurable": {"thread_id": "demo"}}

    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "List files in /tmp"}]},
        config,
    )
    for msg in result["messages"]:
        print(f"{msg.type}: {msg.content}")

asyncio.run(main())
```
