# 05 — Tools

Tools are the building blocks agents use to interact with the world. In LangChain v1, every tool is a `BaseTool` subclass. The three patterns are: `@tool` decorator (simplest), `StructuredTool.from_function` (explicit schema), and `BaseTool` subclass (stateful, complex).

---

## `@tool` decorator

```python
from langchain_core.tools import tool   # or: from langchain.tools import tool

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers together.

    Args:
        a: First integer.
        b: Second integer.
    """
    return a * b
```

The decorator:
- Infers `name` from the function name (snake_case).
- Infers `description` from the docstring (shown to the model — make it precise).
- Infers `args_schema` from type hints (generates a Pydantic model automatically).
- Wraps errors in `ToolException` by default if `handle_tool_error=True`.

### What the model sees

```python
print(multiply.name)          # "multiply"
print(multiply.description)   # "Multiply two integers together.\n\nArgs:\n..."
print(multiply.args)          # {"a": {"type": "integer", "description": "..."}, ...}
```

The description is the model's primary signal for deciding whether/when to call this tool. Write it from the model's perspective: what does this do, when should I use it?

### Async tools

```python
@tool
async def async_search(query: str) -> str:
    """Search an external API asynchronously."""
    async with aiohttp.ClientSession() as session:
        async with session.get(f"https://api.example.com/search?q={query}") as r:
            return await r.text()
```

LangGraph's `ToolNode` and `create_agent` both support async tools.

### `return_direct=True`

Returns the tool's output directly as the final agent response (skipping further model calls):

```python
@tool(return_direct=True)
def final_answer(text: str) -> str:
    """Emit the final answer to the user."""
    return text
```

### `parse_docstring=True` / `docstring_format`

For NumPy or Google-style docstrings:

```python
@tool(parse_docstring=True)
def add(a: int, b: int) -> int:
    """Add two numbers.

    Args:
        a (int): First number.
        b (int): Second number.

    Returns:
        int: Sum.
    """
    return a + b
```

### `args_schema` override

Pass a custom Pydantic model to override auto-inferred schema:

```python
from pydantic import BaseModel, Field

class SearchInput(BaseModel):
    query: str = Field(description="The search query. Be specific.")
    max_results: int = Field(default=5, ge=1, le=20,
                             description="Number of results to return (1-20).")
    language: str = Field(default="en", pattern="^[a-z]{2}$",
                          description="ISO 639-1 language code.")

@tool(args_schema=SearchInput)
def search(query: str, max_results: int = 5, language: str = "en") -> list[dict]:
    """Search for information. Returns a list of result objects."""
    ...
```

Prefer `args_schema` when you need Pydantic validators (ranges, regex patterns, cross-field validation).

---

## `StructuredTool.from_function`

Explicit construction — useful when you want to separate the function logic from the tool definition, or when creating tools dynamically:

```python
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

class CalculatorInput(BaseModel):
    a: float = Field(description="First operand")
    b: float = Field(description="Second operand")
    op: str = Field(description="Operation: 'add', 'sub', 'mul', 'div'")

def calculator(a: float, b: float, op: str) -> float:
    match op:
        case "add": return a + b
        case "sub": return a - b
        case "mul": return a * b
        case "div": return a / b if b != 0 else float("inf")
        case _: raise ValueError(f"Unknown op: {op}")

async def async_calculator(a: float, b: float, op: str) -> float:
    return calculator(a, b, op)  # in practice, async I/O here

calc_tool = StructuredTool.from_function(
    func=calculator,
    coroutine=async_calculator,  # optional async implementation
    name="calculator",
    description="Perform arithmetic. Use for any math calculation.",
    args_schema=CalculatorInput,
    return_direct=False,
    handle_tool_error=True,       # catch exceptions and return error message
)
```

### Dynamic tool creation (from a list of specs)

```python
def make_api_tool(endpoint: str, description: str) -> StructuredTool:
    class InputSchema(BaseModel):
        payload: str = Field(description="JSON payload string")

    def call_api(payload: str) -> str:
        import httpx
        r = httpx.post(endpoint, json={"data": payload})
        return r.text

    return StructuredTool.from_function(
        func=call_api,
        name=endpoint.split("/")[-1].replace("-", "_"),
        description=description,
        args_schema=InputSchema,
    )

tools = [make_api_tool(ep, desc) for ep, desc in api_specs]
```

---

## `BaseTool` subclass — stateful tools

When a tool needs initialization (an HTTP client, a DB connection, credentials), subclass `BaseTool`:

```python
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
import httpx

class SearchInput(BaseModel):
    query: str = Field(description="The search query")
    k: int = Field(default=5, description="Number of results")

class SearchTool(BaseTool):
    name: str = "web_search"
    description: str = "Search the web for current information."
    args_schema: type[BaseModel] = SearchInput

    # Stateful fields — initialized once, reused across calls
    api_key: str
    _client: httpx.Client | None = None

    def model_post_init(self, __context):
        self._client = httpx.Client(headers={"X-API-Key": self.api_key})

    def _run(self, query: str, k: int = 5) -> str:
        """Sync implementation — must define this."""
        r = self._client.get("/search", params={"q": query, "k": k})
        return r.text

    async def _arun(self, query: str, k: int = 5) -> str:
        """Async implementation — optional but recommended."""
        async with httpx.AsyncClient(headers={"X-API-Key": self.api_key}) as client:
            r = await client.get("/search", params={"q": query, "k": k})
            return r.text

    def _handle_error(self, error: Exception) -> str:
        return f"Search failed: {error}"

# Instantiate:
search_tool = SearchTool(api_key="sk-...")
```

**`handle_tool_error`:**
```python
# On @tool or StructuredTool, set handle_tool_error:
@tool(handle_tool_error=True)      # returns exception message as string
def risky_tool(x: str) -> str: ...

@tool(handle_tool_error="Tool failed. Try rephrasing.")  # custom message
def risky_tool2(x: str) -> str: ...

@tool(handle_tool_error=lambda e: f"Error: {e}")   # callable
def risky_tool3(x: str) -> str: ...
```

---

## Tool calling with Gemini — how it works

Gemini (via `ChatGoogleGenerativeAI.bind_tools`) converts tool definitions to Gemini function declarations and sends them with every model call. The model responds either with normal text OR with `tool_calls` (a list of `{"name": ..., "args": {...}, "id": ...}` dicts on the `AIMessage`).

```python
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, ToolMessage

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
tools = [search_tool, calc_tool]
llm_with_tools = llm.bind_tools(tools)

# Step 1: model decides what to call
ai_msg = llm_with_tools.invoke([HumanMessage(content="What's 42 * 18?")])
print(ai_msg.tool_calls)
# [{"name": "calculator", "args": {"a": 42, "b": 18, "op": "mul"}, "id": "call_1"}]

# Step 2: execute the tool manually (if not using an agent)
for tc in ai_msg.tool_calls:
    tool_fn = {t.name: t for t in tools}[tc["name"]]
    result = tool_fn.invoke(tc["args"])
    tool_msg = ToolMessage(content=str(result), tool_call_id=tc["id"])

# Step 3: send tool result back
final = llm_with_tools.invoke([
    HumanMessage(content="What's 42 * 18?"),
    ai_msg,
    tool_msg,
])
print(final.content)  # "42 × 18 = 756"
```

**In practice**: agents (via `create_agent` or `create_react_agent`) handle this loop automatically — you only write the tool function, not the orchestration.

### `tool_choice` parameter

Force the model to call a specific tool or any tool:

```python
# Force use of a specific tool:
llm_forced = llm.bind_tools(tools, tool_choice="calculator")

# Force use of any tool (model must call at least one):
llm_forced_any = llm.bind_tools(tools, tool_choice="any")

# Default (model decides): no tool_choice kwarg
```

---

## Injecting runtime context into tools (LangGraph)

In LangGraph, tools can access the state, config, and store via injected parameters. These are populated by `ToolNode` automatically and are **not part of the tool's LLM-visible schema**:

```python
from typing import Annotated
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import InjectedState, InjectedStore
from langgraph.store.base import BaseStore

@tool
def get_user_preference(
    preference_key: str,
    # Injected — NOT in the LLM schema:
    config: Annotated[RunnableConfig, InjectedConfig()],
    store: Annotated[BaseStore, InjectedStore()],
) -> str:
    """Look up a user preference by key."""
    user_id = config["configurable"].get("user_id", "anon")
    ns = ("user_prefs", user_id)
    item = store.get(ns, preference_key)
    return item.value["text"] if item else "Not set"
```

Use `InjectedState()` to access the full LangGraph state:

```python
from langgraph.prebuilt import InjectedState
from langgraph.graph import MessagesState

@tool
def summarize_history(
    topic: str,
    state: Annotated[MessagesState, InjectedState()],
) -> str:
    """Summarize conversation history about a topic."""
    msgs = [m.content for m in state["messages"] if hasattr(m, "content")]
    return f"History has {len(msgs)} messages about {topic}."
```

---

## Tool best practices

1. **Name tools in snake_case.** Gemini and other models handle `web_search`, not `WebSearch` or `web-search`.

2. **Write descriptive docstrings.** The model reads the description to decide when to call the tool. Be specific: include constraints, input formats, and when NOT to use it.

3. **Validate with Pydantic.** Always define `args_schema` with `Field` descriptions. This gives the model field-level guidance and prevents nonsense inputs.

4. **Keep tools small and focused.** A tool that does one thing well is better than a multi-function monolith. The model will compose them.

5. **Return strings.** Tool outputs must be serializable to `ToolMessage.content`. Return `str` or serialize to JSON. `ToolNode` auto-converts non-string results via `str(result)` — make sure that's useful output.

6. **Handle errors.** Use `handle_tool_error=True` or implement `_handle_error`. Unhandled tool exceptions crash the agent loop in `ToolNode` by default.

7. **Avoid side-effects in `args_schema` validators.** They're called every time the model's tool call is parsed — keep them pure validation logic.

8. **For async agents, implement `_arun`.** `ToolNode` in async mode calls `_arun` when available; falls back to running `_run` in a thread executor. Implement `_arun` for any I/O-heavy tool.

9. **Don't put secrets in tool names or descriptions.** The tool definition is sent to the model (and logged to LangSmith). Keep API keys in the tool's state, not its metadata.
