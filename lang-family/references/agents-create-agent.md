# 06 — `langchain.agents.create_agent` (LangChain v1 Primary Agent Factory)

`create_agent` is the canonical agent factory in LangChain v1. It returns a **compiled LangGraph graph** that runs a model→tools loop (ReAct style) until the model stops calling tools or a limit is hit. It sits on top of `langgraph.prebuilt.create_react_agent` and adds the middleware system.

---

## Import

```python
from langchain.agents import create_agent
```

---

## Full signature

```python
agent = create_agent(
    # ── Required ──────────────────────────────────────────────────
    model,                       # BaseChatModel | str  (e.g. "google_genai:gemini-2.5-flash")
    tools,                       # list[BaseTool | Callable]

    # ── System prompt ─────────────────────────────────────────────
    system_prompt=None,          # str | None  (plain system prompt)
    prompt=None,                 # ChatPromptTemplate | None  (full prompt override)

    # ── Structured final output ───────────────────────────────────
    response_format=None,        # type[BaseModel] | ProviderStrategy | ToolStrategy | None

    # ── Middleware ────────────────────────────────────────────────
    middleware=None,             # list[AgentMiddleware] | None

    # ── State customization ───────────────────────────────────────
    state_schema=None,           # TypedDict class ONLY (not Pydantic or dataclass in v1)
    context_schema=None,         # TypedDict class for run-time context (not persisted)

    # ── Persistence ───────────────────────────────────────────────
    checkpointer=None,           # BaseCheckpointSaver | None
    store=None,                  # BaseStore | None

    # ── Execution control ─────────────────────────────────────────
    interrupt_before=None,       # list[str]: node names to pause before
    interrupt_after=None,        # list[str]: node names to pause after
    debug=False,                 # bool: verbose execution logging
    name="agent",                # str: graph name (affects LangSmith trace)
)
```

Returns: `CompiledStateGraph` — implements the full `Runnable` interface (`invoke`, `ainvoke`, `stream`, `astream`, `astream_events`, `batch`, `abatch`).

---

## Minimal example (Gemini)

```python
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

@tool
def get_current_time(timezone: str = "UTC") -> str:
    """Return the current time in the given timezone."""
    from datetime import datetime, timezone as tz
    import zoneinfo
    try:
        return datetime.now(zoneinfo.ZoneInfo(timezone)).strftime("%H:%M:%S %Z")
    except Exception:
        return datetime.now(tz.utc).strftime("%H:%M:%S UTC")

agent = create_agent(
    model=ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2),
    tools=[get_current_time],
    system_prompt="You are a helpful assistant. Use tools when needed.",
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "What time is it in Tokyo?"}]
})
print(result["messages"][-1].content)
```

---

## Input format

The graph always takes a dict with a `"messages"` key (and any extra keys if you have a custom `state_schema`):

```python
# Shorthand (HumanMessage is created automatically from dict):
agent.invoke({"messages": [{"role": "user", "content": "..."}]})

# Full message objects:
from langchain_core.messages import HumanMessage
agent.invoke({"messages": [HumanMessage(content="...")]})

# With extra state (when using state_schema):
agent.invoke({"messages": [...], "user_id": "u123", "preferences": {}})
```

---

## Output format

The output dict has the same shape as the input state plus any updates:

```python
result = agent.invoke(...)
result["messages"]              # full message list including AI and tool messages
result["messages"][-1].content  # final AI response text

# If response_format was set:
result["structured_response"]   # the parsed Pydantic object
```

---

## `system_prompt` vs `prompt`

```python
# Simple string system prompt (most common):
agent = create_agent(
    model=llm, tools=tools,
    system_prompt="You are an expert {domain} assistant.",
)

# Full ChatPromptTemplate (when you need MessagesPlaceholder, few-shot, or
# more control over the prompt structure):
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a {domain} expert."),
    ("system", "Always cite your sources."),
    MessagesPlaceholder("messages"),    # REQUIRED: the agent injects messages here
])
agent = create_agent(model=llm, tools=tools, prompt=prompt)
agent.invoke({"messages": [...], "domain": "finance"})
```

**Gotcha:** if you pass `prompt=ChatPromptTemplate`, it MUST include `MessagesPlaceholder("messages")`. The agent injects the conversation messages at this slot.

---

## `response_format` — structured final output

The agent's final response (after the last model call) is parsed into a structured schema. The model is asked to produce this at the end of execution.

```python
from pydantic import BaseModel, Field

class TravelPlan(BaseModel):
    destination: str
    duration_days: int
    budget_usd: float
    highlights: list[str] = Field(default_factory=list)

# Provider strategy (default when supported — uses native json_schema):
from langchain.agents import ProviderStrategy, ToolStrategy

agent = create_agent(
    model=llm, tools=tools,
    response_format=TravelPlan,           # shorthand — auto-selects strategy
)
# or explicit:
agent = create_agent(
    model=llm, tools=tools,
    response_format=ProviderStrategy(TravelPlan),  # forces native JSON schema
)
# or force function-calling (universal fallback):
agent = create_agent(
    model=llm, tools=tools,
    response_format=ToolStrategy(TravelPlan),      # uses a hidden "respond" tool
)

result = agent.invoke({"messages": [{"role": "user", "content": "Plan a trip to Tokyo."}]})
plan: TravelPlan = result["structured_response"]
print(plan.destination)   # "Tokyo"
```

When `response_format` is set, the agent runs its normal tool loop then makes one final call to produce structured output.

---

## `state_schema` — custom state fields

Extend the agent state with extra fields. **Must be a `TypedDict` in v1** (not Pydantic, not dataclass):

```python
from typing import Annotated, Optional
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage

class CustomerAgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]  # required
    user_id: str
    account_tier: Optional[str]
    session_context: dict

agent = create_agent(
    model=llm, tools=tools,
    state_schema=CustomerAgentState,
    system_prompt="You are a customer support agent.",
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "I need help with my order."}],
    "user_id": "cust_123",
    "account_tier": "premium",
    "session_context": {},
})
```

Tools can read extra state via `InjectedState`:

```python
from typing import Annotated
from langgraph.prebuilt import InjectedState

@tool
def check_account_status(
    query: str,
    state: Annotated[CustomerAgentState, InjectedState()],
) -> str:
    """Check the account status for the current user."""
    return f"User {state['user_id']} is {state['account_tier']} tier."
```

---

## `context_schema` — run-time context (not persisted)

Context is like `state_schema` but is NOT checkpointed. Use for per-invocation config that doesn't need to survive restarts:

```python
class AgentContext(TypedDict):
    current_user_timezone: str
    feature_flags: dict

agent = create_agent(
    model=llm, tools=tools,
    context_schema=AgentContext,
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "What time is it?"}]},
    config={"configurable": {
        "thread_id": "t1",
        "current_user_timezone": "Asia/Tokyo",
        "feature_flags": {"new_ui": True},
    }},
)
```

Access context inside a node or tool via `runtime.context` (using the `Runtime` API):

```python
from langgraph.runtime import Runtime

@tool
def get_local_time(state: Annotated[dict, InjectedState()],
                   runtime: Annotated[Runtime, ...]) -> str:
    """Get the current time in the user's timezone."""
    tz = runtime.context.get("current_user_timezone", "UTC")
    ...
```

---

## With checkpointer (persistent memory, HITL)

```python
from langgraph.checkpoint.sqlite import SqliteSaver

with SqliteSaver.from_conn_string("agent.sqlite") as ckpt:
    agent = create_agent(
        model=llm, tools=tools,
        system_prompt="You are a helpful assistant.",
        checkpointer=ckpt,
    )

    # First turn:
    cfg = {"configurable": {"thread_id": "user-42"}}
    result1 = agent.invoke(
        {"messages": [{"role": "user", "content": "My name is Alice."}]},
        config=cfg,
    )

    # Second turn — history is automatically loaded from checkpointer:
    result2 = agent.invoke(
        {"messages": [{"role": "user", "content": "What's my name?"}]},
        config=cfg,
    )
    print(result2["messages"][-1].content)  # "Your name is Alice."
```

---

## Streaming

```python
# Streaming token output:
for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "Explain quantum computing."}]},
    stream_mode="messages",
):
    msg, metadata = chunk
    if hasattr(msg, "content") and msg.content:
        print(msg.content, end="", flush=True)

# Streaming state updates:
for chunk in agent.stream(input, stream_mode="updates"):
    for node_name, update in chunk.items():
        print(f"Node {node_name} produced: {list(update.keys())}")

# Combined (v2 unified format):
for chunk in agent.stream(input, stream_mode=["messages", "updates"], version="v2"):
    if chunk["type"] == "messages":
        msg, _ = chunk["data"]
        if msg.content: print(msg.content, end="")
```

---

## `interrupt_before` / `interrupt_after` — static breakpoints

Pause execution before/after specific nodes. Requires a checkpointer.

```python
agent = create_agent(
    model=llm, tools=tools,
    checkpointer=ckpt,
    interrupt_before=["tools"],   # pause before tool execution
)

cfg = {"configurable": {"thread_id": "t1"}}
out = agent.invoke(input, cfg)
# If model requested a tool call, execution pauses here.
# out["__interrupt__"] will contain an Interrupt object.

# Inspect the pending tool call:
state = agent.get_state(cfg)
print(state.values["messages"][-1].tool_calls)  # see what tool was requested

# Resume (approve execution):
from langgraph.types import Command
result = agent.invoke(Command(resume=True), cfg)

# Or reject by editing state:
agent.update_state(cfg, {"messages": [AIMessage(content="I'll skip the tool call.")]})
result = agent.invoke(None, cfg)
```

---

## How `create_agent` relates to LangGraph

Under the hood, `create_agent` builds and compiles a `StateGraph` that looks like:

```
START → agent_node → (tool_calls?) → tools_node → agent_node → ... → END
                           └── (no tool calls) → END
```

It's equivalent to the standard `create_react_agent` from `langgraph.prebuilt` plus the middleware layer. If you need to:
- Add extra nodes or edges → build a `StateGraph` directly (see `12-langgraph-stategraph.md`).
- Use finer-grained LangGraph features → use `langgraph.prebuilt.create_react_agent`.
- Add lifecycle hooks but stay in `create_agent` → use middleware (see `07-middleware.md`).

---

## Gotchas

1. **`state_schema` must be `TypedDict`** in v1. Pydantic and dataclasses were removed from `create_agent`. `StateGraph` itself still accepts them.

2. **`prompt` with `MessagesPlaceholder` is required when using a `ChatPromptTemplate`.** If you omit `MessagesPlaceholder("messages")`, the agent messages won't be injected and the model won't see the conversation.

3. **`response_format` makes an extra LLM call.** The final structured-output call happens after the tool loop. Budget for 1 extra LLM round-trip.

4. **`thread_id` is required in config when a checkpointer is set.** Without it, the checkpointer doesn't know which thread to read/write. You'll get an error.

5. **`interrupt_before=["tools"]` pauses on every tool call.** If the model makes 3 tool calls in a single turn, you'll get 3 interrupts. Handle this with `HumanInTheLoopMiddleware` for more ergonomic approval flows.

6. **The agent loop has a default `recursion_limit=25`.** Override per-invocation in config: `{"recursion_limit": 50}`. Infinite loops in agent prompts will hit this limit and raise `GraphRecursionError`.
