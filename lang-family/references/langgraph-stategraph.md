# 12 — LangGraph `StateGraph` (Core)

## What is LangGraph?

LangGraph is a graph-based orchestration runtime built on top of LangChain. It adds:
- **Shared, typed state** with per-field reducers (not just a dict flowing through steps)
- **Cycles** (loops, reflection, retry loops — impossible in pure LCEL)
- **Durable execution** (checkpointers that survive crashes)
- **Human-in-the-loop** (interrupt any node, wait for human, resume)
- **Native streaming** at multiple granularities (tokens, node updates, full state)
- **Multi-agent** (subgraphs, supervisor, swarm)

Use LangGraph whenever your workflow has: state that persists across turns, conditional branching back to earlier nodes, parallel fan-out/fan-in, human approval steps, or orchestration of multiple agents.

---

## `StateGraph` constructor

```python
from langgraph.graph import StateGraph

graph_builder = StateGraph(
    state_schema,           # TypedDict | Pydantic BaseModel | dataclass (ALL SUPPORTED here,
                            # unlike create_agent where only TypedDict is allowed)
    context_schema=None,    # TypedDict for per-run context (not persisted, replaces config_schema)
    input_schema=None,      # TypedDict: restrict what the caller can pass in
    output_schema=None,     # TypedDict: restrict what the graph returns
)
```

## Defining state

State is a **typed shared dict** that every node reads and (partially) writes. Use `Annotated[T, reducer]` to declare reducers per field.

### TypedDict state (most common)

```python
from typing import Annotated, Optional
from typing_extensions import TypedDict
from operator import add
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    # Messages: use add_messages reducer — handles ID-based merge and
    # string-to-message coercion:
    messages: Annotated[list[AnyMessage], add_messages]

    # Accumulate items (list concatenation):
    retrieved_chunks: Annotated[list[str], add]

    # Overwrite on each update (no reducer):
    current_query: str
    status: str

    # Optional field:
    error: Optional[str]

    # Counter (custom reducer):
    attempt_count: Annotated[int, lambda a, b: a + b]
```

### Pydantic state (type validation + defaults)

```python
from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages

class State(BaseModel):
    messages: Annotated[list[AnyMessage], add_messages] = Field(default_factory=list)
    user_id: str = ""
    context: dict = Field(default_factory=dict)
```

### `MessagesState` — prebuilt messages-only state

```python
from langgraph.graph import MessagesState
# MessagesState is equivalent to TypedDict with:
#   messages: Annotated[list[AnyMessage], add_messages]
# Inherit to add more fields:
class State(MessagesState):
    user_id: str
    retrieved_docs: list
```

### `add_messages` reducer semantics

`add_messages` is NOT a simple list append. It:
1. **Appends** messages that have a new ID (or no ID → assigned one).
2. **Replaces** existing messages by ID if the same ID appears again. This is how `ToolMessage` responses update the state — they match the `tool_call_id`.
3. Converts `(role, content)` tuples and dicts to message objects automatically.
4. Handles `RemoveMessage` objects (delete a message by ID).

```python
from langchain_core.messages import RemoveMessage

# Delete a message by ID:
def cleanup_node(state: State) -> dict:
    msg_to_delete = state["messages"][0]
    return {"messages": [RemoveMessage(id=msg_to_delete.id)]}
```

---

## Nodes

Nodes are Python functions (sync or async) that:
- Take the full state (or a subset via `InjectedState`)
- Optionally take a `config: RunnableConfig` second arg
- Optionally take a `runtime: Runtime` second arg (preferred in v1 for accessing store/context)
- Return a **partial state dict** (only the keys they're updating — missing keys are unchanged)

```python
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime

# Minimal:
def my_node(state: AgentState) -> dict:
    return {"status": "done"}

# With config:
def my_node_with_config(state: AgentState, config: RunnableConfig) -> dict:
    user_id = config.get("configurable", {}).get("user_id")
    return {"user_id": user_id}

# With runtime (access store, context):
def my_node_with_runtime(state: AgentState, runtime: Runtime) -> dict:
    # runtime.store — the BaseStore (if set on compile)
    # runtime.context — the context_schema dict (if set)
    item = runtime.store.get(("namespace",), "key")
    return {"status": "fetched"}

# Async node:
async def async_node(state: AgentState) -> dict:
    result = await some_async_call(state["current_query"])
    return {"retrieved_chunks": [result]}
```

**What nodes must NOT do:**
- Modify the state dict in-place — always return a new dict.
- Access any global mutable state without thread safety.

**Node return value semantics:**
- `return {}` — no state changes.
- `return {"key": value}` — update only `key`; other keys unchanged.
- `return {"messages": [ai_msg]}` — if `messages` has `add_messages` reducer, this *appends* `ai_msg`.
- `return Command(update={...}, goto="next_node")` — update state AND control routing (see `13-langgraph-control-flow.md`).

---

## Building and compiling the graph

All `add_node`, `add_edge`, etc. methods are **chainable** (return `self`):

```python
from langgraph.graph import StateGraph, START, END

builder = StateGraph(AgentState)

# Add nodes:
builder.add_node("fetch", fetch_node)
builder.add_node("process", process_node)
builder.add_node("respond", respond_node)

# Add edges:
builder.add_edge(START, "fetch")          # START is the entry point
builder.add_edge("fetch", "process")
builder.add_edge("process", "respond")
builder.add_edge("respond", END)           # END is the exit point

# Or chained:
graph = (
    StateGraph(AgentState)
    .add_node("fetch", fetch_node)
    .add_node("process", process_node)
    .add_node("respond", respond_node)
    .add_edge(START, "fetch")
    .add_edge("fetch", "process")
    .add_edge("process", "respond")
    .add_edge("respond", END)
    .compile()
)
```

### `compile()` parameters

```python
compiled = builder.compile(
    checkpointer=None,           # BaseCheckpointSaver | None: enables persistence
    store=None,                  # BaseStore | None: enables long-term memory
    interrupt_before=[],         # list[str]: always pause before these node names
    interrupt_after=[],          # list[str]: always pause after these node names
    debug=False,                 # bool: verbose execution logs
    name="my_graph",             # str: name for LangSmith traces and subgraph use
)
```

---

## Invoking the compiled graph

The compiled graph is a `Runnable`. Every invocation needs an initial state (or partial state):

```python
from langchain_core.messages import HumanMessage

# Invoke (blocks until completion):
result: dict = graph.invoke(
    {"messages": [HumanMessage(content="Hello!")]},
    config={"configurable": {"thread_id": "t1"}},
)
print(result["messages"][-1].content)

# Stream updates (one dict per node completion):
for chunk in graph.stream(
    {"messages": [HumanMessage(content="Hello!")]},
    config={"configurable": {"thread_id": "t1"}},
    stream_mode="updates",
):
    for node, delta in chunk.items():
        print(f"{node}: {delta}")

# Async:
result = await graph.ainvoke(...)
async for chunk in graph.astream(...): ...
```

### `recursion_limit`

Prevents infinite loops. Default is 25. Override in config:

```python
config = {
    "configurable": {"thread_id": "t1"},
    "recursion_limit": 50,       # allow more hops in complex graphs
}
```

Each node traversal counts as 1 step. A graph with a 10-node loop would need `recursion_limit >= 10` per loop iteration.

---

## `Runtime` API (LangGraph v1)

In v1, the preferred way to access per-run context and the store inside nodes is via the `Runtime` parameter:

```python
from langgraph.runtime import Runtime

def node_with_store(state: AgentState, runtime: Runtime) -> dict:
    # Access the store (BaseStore):
    items = runtime.store.search(("namespace",), query=state["current_query"])

    # Access per-run context (from context_schema):
    user_tz = runtime.context.get("user_timezone", "UTC")

    return {"status": "ok"}
```

Alternatively, use `get_store()` and `get_config()` from `langgraph.config`:

```python
from langgraph.config import get_store, get_config

def another_node(state: AgentState) -> dict:
    store = get_store()      # returns BaseStore or None
    config = get_config()    # returns RunnableConfig
    return {}
```

---

## Graph introspection

```python
# Get the Mermaid diagram:
graph.get_graph().draw_mermaid()         # returns Mermaid syntax string
graph.get_graph().draw_mermaid_png()     # returns PNG bytes

# Print nodes and edges:
print(graph.get_graph().nodes)
print(graph.get_graph().edges)

# For subgraphs, expand them:
graph.get_graph(xray=True).draw_mermaid()
```

---

## Example: simple multi-step pipeline with a conditional loop

```python
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    attempts: Annotated[int, lambda a, b: a + b]
    answer: str

def research(state: State) -> dict:
    ai = llm.invoke(state["messages"])
    return {"messages": [ai], "attempts": 1}

def check_quality(state: State) -> str:
    last = state["messages"][-1].content
    if len(last) < 50 and state["attempts"] < 3:
        return "retry"
    return "done"

def finalize(state: State) -> dict:
    return {"answer": state["messages"][-1].content}

graph = (
    StateGraph(State)
    .add_node("research", research)
    .add_node("finalize", finalize)
    .add_edge(START, "research")
    .add_conditional_edges("research", check_quality, {
        "retry": "research",   # loop back
        "done": "finalize",
    })
    .add_edge("finalize", END)
    .compile()
)

result = graph.invoke({
    "messages": [HumanMessage(content="Explain RAG in one sentence.")],
    "attempts": 0,
    "answer": "",
})
print(result["answer"])
```

---

## `input_schema` and `output_schema`

Restrict what callers can pass in, and what the graph returns:

```python
class InputSchema(TypedDict):
    question: str        # callers only need to pass question
    user_id: str

class OutputSchema(TypedDict):
    answer: str          # graph only returns answer, not internal state

graph = StateGraph(
    FullState,
    input_schema=InputSchema,
    output_schema=OutputSchema,
).add_node(...).compile()

result: OutputSchema = graph.invoke({"question": "...", "user_id": "u1"})
# result only has {"answer": "..."}
```

---

## Gotchas

1. **Reducers are MANDATORY for multi-writer channels.** If two nodes write to the same key without a reducer, LangGraph raises `InvalidUpdateError`. When in doubt, use `Annotated[list[str], add]` or a custom reducer.

2. **Nodes must return a dict, not the full state.** Return only the keys you're updating. Returning the full state object can cause unexpected reducer behavior.

3. **`add_messages` replaces by message ID, not by position.** If you generate a new `AIMessage` with the same `id` as an existing one, it replaces (not appends). Let message IDs be auto-assigned unless you explicitly need overwrite semantics.

4. **`START` and `END` are sentinel strings** from `langgraph.graph`. Don't confuse them with `"__start__"` or `"__end__"` — those are internal node names.

5. **Async nodes in a sync graph work**, but if you call `graph.invoke` (sync), async nodes are run via `asyncio.run()` internally. For proper async, use `await graph.ainvoke(...)`.

6. **`context_schema` (not `config_schema`).** The `config_schema` parameter was deprecated in v0.6 and removed in v2.0. Use `context_schema` and access values via `runtime.context` or `get_config()["configurable"]`.
