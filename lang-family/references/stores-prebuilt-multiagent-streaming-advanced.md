# 17 — LangGraph Stores (Long-Term, Cross-Thread Memory)

Checkpointers = per-thread, short-term state.
Stores = cross-thread, long-term memory. Data persists across different conversation threads and users.

---

## Store interface

```python
from langgraph.store.base import BaseStore

store.put(namespace: tuple, key: str, value: dict) -> None
store.get(namespace: tuple, key: str) -> Item | None
store.delete(namespace: tuple, key: str) -> None
store.search(namespace: tuple, query: str, limit: int, filter: dict) -> list[SearchItem]

# Async:
await store.aput(...)
await store.aget(...)
await store.adelete(...)
results = await store.asearch(...)

# Batch operations:
store.batch([PutOp(...), GetOp(...), SearchOp(...)])
```

`namespace` is a tuple of strings acting as a hierarchical key: `("user_123", "memories")`, `("global", "facts")`, etc. This allows per-user, per-agent, or per-concept namespacing.

`Item` fields: `.namespace`, `.key`, `.value` (the dict you stored), `.created_at`, `.updated_at`.

---

## `InMemoryStore` (dev)

```python
from langgraph.store.memory import InMemoryStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Without embeddings (key-value only):
store = InMemoryStore()

# With embeddings for semantic search:
store = InMemoryStore(
    index={
        "dims": 768,                  # must match embedding output dimension
        "embed": GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-001",
            task_type="SEMANTIC_SIMILARITY",
            output_dimensionality=768,
        ),
        "fields": ["text", "summary"],  # which keys in the stored value to embed
    }
)
```

---

## `PostgresStore` (production)

```python
from langgraph.store.postgres import PostgresStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings

store_cm = PostgresStore.from_conn_string(
    "postgresql://user:pass@host:5432/db",
    index={
        "dims": 768,
        "embed": GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-001",
            task_type="SEMANTIC_SIMILARITY",
            output_dimensionality=768,
        ),
        "fields": ["text"],
    },
)

with store_cm as store:
    store.setup()   # run migrations once
    graph = builder.compile(checkpointer=ckpt, store=store)
```

---

## Accessing the store inside nodes

**Via `Runtime` (preferred in v1):**
```python
from langgraph.runtime import Runtime

def memory_node(state: State, runtime: Runtime) -> dict:
    ns = ("user", state["user_id"], "memories")

    # Store a fact:
    runtime.store.put(ns, "name", {"text": "User's name is Alice."})

    # Exact lookup:
    item = runtime.store.get(ns, "name")
    if item:
        print(item.value["text"])

    # Semantic search:
    results = runtime.store.search(
        ns,
        query=state["messages"][-1].content,
        limit=5,
        filter={"category": "preference"},  # optional metadata filter
    )
    memories = [r.value["text"] for r in results]
    return {"context": memories}
```

**Via `get_store()` (functional alternative):**
```python
from langgraph.config import get_store

def node(state):
    store = get_store()
    store.put(("global",), "fact", {"text": "Important fact."})
    return {}
```

**In tools (via `InjectedStore`):**
```python
from typing import Annotated
from langgraph.prebuilt import InjectedStore
from langgraph.store.base import BaseStore
from langchain_core.tools import tool

@tool
def remember(
    fact: str,
    store: Annotated[BaseStore, InjectedStore()],
) -> str:
    """Store a fact for later retrieval."""
    store.put(("memories",), f"fact_{hash(fact)}", {"text": fact})
    return "Remembered."
```

---

## Memory design patterns

### Semantic memory (facts about the world / user)
```python
runtime.store.put(("user", user_id, "facts"), key, {
    "text": "User prefers Python over JavaScript.",
    "category": "preference",
    "source": "explicit",
})
# Retrieve relevant facts:
facts = runtime.store.search(("user", user_id, "facts"),
                             query=current_message, limit=5)
```

### Episodic memory (past interaction summaries)
```python
runtime.store.put(("user", user_id, "episodes"), f"ep_{timestamp}", {
    "text": summary_of_past_session,
    "date": timestamp,
    "outcome": "success",
})
```

### Procedural memory (evolving system prompts / instructions)
```python
runtime.store.put(("agent", agent_id, "instructions"), "system_prompt", {
    "text": "You are a coding assistant who prefers concise answers.",
    "updated_at": datetime.now().isoformat(),
})
```

---

## `langmem` library (higher-level memory helpers)

`langmem` wraps the Store API with higher-level tools and a background memory manager:

```python
# pip install langmem
from langmem import create_manage_memory_tool, create_search_memory_tool

manage_memory = create_manage_memory_tool(namespace=("user", "{user_id}", "memories"))
search_memory = create_search_memory_tool(namespace=("user", "{user_id}", "memories"))

# Use as tools in an agent:
agent = create_agent(model=llm, tools=[manage_memory, search_memory, ...])
```

`langmem` also provides a background `MemoryManager` that automatically extracts and updates memories from conversation history.

---

# 18 — LangGraph Prebuilt Agents

## `langgraph.prebuilt.create_react_agent`

The lower-level prebuilt agent. Returns a `CompiledStateGraph` implementing the standard model→tools→model loop. Use this when you want graph-level customization without the full middleware framework of `langchain.agents.create_agent`.

```python
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI

agent = create_react_agent(
    model=ChatGoogleGenerativeAI(model="gemini-2.5-flash"),
    tools=tools,

    # Prompt:
    prompt="You are helpful.",            # str | SystemMessage | callable | Runnable
    # OR as callable (runs before each model call):
    # prompt=lambda state: [SystemMessage(content="...")] + state["messages"],

    # Structured output:
    response_format=None,                 # type[BaseModel] | (prompt, schema) | None

    # State:
    state_schema=None,                    # TypedDict | Pydantic | dataclass
    state_modifier=None,                  # DEPRECATED — use prompt instead

    # Persistence:
    checkpointer=None,
    store=None,

    # Hooks:
    pre_model_hook=None,                  # callable(state) -> dict, runs before each model call
    post_model_hook=None,                 # callable(state) -> dict, runs after each model call

    # Execution:
    interrupt_before=None,                # list[str]
    interrupt_after=None,                 # list[str]
    debug=False,
    version="v2",                         # "v1" | "v2": v2 distributes tool calls via Send
    name="agent",
)
```

**`version="v2"`** (default): when the model returns multiple tool calls, they're dispatched in parallel via the `Send` API. `version="v1"` runs them sequentially.

**`pre_model_hook` and `post_model_hook`:** run before/after each model call; can modify state. Lighter-weight than middleware, but only two hooks (no tool wrapping, no agent lifecycle).

---

## `ToolNode`

Executes tool calls from the last `AIMessage` in state. Parallel by default for multi-call responses.

```python
from langgraph.prebuilt import ToolNode
from langchain_core.messages import AIMessage

tool_node = ToolNode(
    tools=tools,
    name="tools",                       # node name
    handle_tool_errors=True,            # catch exceptions, return error as ToolMessage
    # handle_tool_errors="Error: {error}" — custom template
    # handle_tool_errors=my_fn          — callable(ToolException) -> str
)

# ToolNode reads state["messages"][-1].tool_calls and executes them.
# Returns: {"messages": [ToolMessage(...), ToolMessage(...)]}
# Pair with tools_condition:
builder.add_conditional_edges("agent", tools_condition)  # → "tools" or END
builder.add_edge("tools", "agent")
```

**Inject state/store into tools:** `ToolNode` automatically populates `InjectedState` and `InjectedStore` parameters on tools (see `05-tools.md`).

---

## `tools_condition`

```python
from langgraph.prebuilt import tools_condition
# Reads state["messages"][-1]
# Returns "tools" if it has tool_calls, else END
builder.add_conditional_edges("agent", tools_condition)
```

---

## When to use what

| Need | Use |
|---|---|
| Quick agent, standard ReAct loop, middleware | `langchain.agents.create_agent` |
| Agent with custom nodes or routing | `langgraph.prebuilt.create_react_agent` |
| Full custom graph (plan-execute, multi-agent) | `StateGraph` directly |
| Just the tool execution node | `ToolNode` |
| Just the routing logic | `tools_condition` |

---

# 19 — LangGraph Multi-Agent Patterns

## Supervisor pattern (`langgraph-supervisor`)

A supervisor LLM routes to specialist workers. More accurate (dedicated routing call), higher latency.

```python
# pip install langgraph-supervisor
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

researcher = create_react_agent(
    model=llm, tools=[search_tool], name="researcher",
    prompt="You search for information. When done, respond with FINAL ANSWER.",
)
writer = create_react_agent(
    model=llm, tools=[draft_tool], name="writer",
    prompt="You write reports. When done, respond with FINAL ANSWER.",
)

supervisor_graph = create_supervisor(
    agents=[researcher, writer],
    model=llm,
    prompt=(
        "You are a supervisor. Route the user's request to the appropriate specialist. "
        "Use 'researcher' for information lookup, 'writer' for drafting content."
    ),
).compile(checkpointer=InMemorySaver())

result = supervisor_graph.invoke(
    {"messages": [{"role": "user", "content": "Research and write a brief on RAG."}]},
    {"configurable": {"thread_id": "t1"}},
)
```

## Swarm pattern (`langgraph-swarm`)

Agents hand off to each other directly via handoff tools. No supervisor. Lower latency (one less LLM call), less predictable routing.

```python
# pip install langgraph-swarm
from langgraph_swarm import create_swarm, create_handoff_tool

# Create handoff tools:
handoff_to_researcher = create_handoff_tool(agent_name="researcher")
handoff_to_writer = create_handoff_tool(agent_name="writer")

researcher = create_react_agent(
    model=llm,
    tools=[search_tool, handoff_to_writer],  # can hand off to writer
    name="researcher",
)
writer = create_react_agent(
    model=llm,
    tools=[draft_tool, handoff_to_researcher],  # can hand off to researcher
    name="writer",
)

swarm = create_swarm([researcher, writer], default_active_agent="researcher")
compiled_swarm = swarm.compile(checkpointer=InMemorySaver())
```

## Hierarchical agents / subgraphs

Compile a subgraph and add it as a node:

```python
# Build subgraph:
sub_graph = (
    StateGraph(SubState)
    .add_node("sub_a", sub_a_node)
    .add_edge(START, "sub_a")
    .add_edge("sub_a", END)
    .compile(name="subgraph_1")   # name helps with LangSmith tracing
)

# Add to parent graph as a node:
parent_graph = (
    StateGraph(ParentState)
    .add_node("sub", sub_graph)   # subgraph is the node function
    .add_edge(START, "sub")
    .add_edge("sub", END)
    .compile(checkpointer=parent_ckpt)
)
```

**State compatibility:** the subgraph's output keys must exist in the parent state and have reducers if multiple subgraph instances write to the same key.

**Cross-graph `Command.PARENT`:** from inside a subgraph node, return `Command(goto="parent_node", graph=Command.PARENT)` to jump to a node in the parent.

## Agent-as-tool

Wrap a compiled agent as a tool callable by another agent:

```python
from langchain_core.tools import tool

research_agent = create_react_agent(model=llm, tools=[search_tool])

@tool
def research(query: str) -> str:
    """Research a topic thoroughly. Returns a detailed summary."""
    result = research_agent.invoke({"messages": [{"role": "user", "content": query}]})
    return result["messages"][-1].content

orchestrator = create_agent(model=llm, tools=[research, code_tool], ...)
```

## Handoff pattern (without `langgraph-swarm`)

Return a `Command` with `goto` set to another agent's node name:

```python
def agent_a_node(state: State) -> Command:
    if "write" in state["messages"][-1].content.lower():
        return Command(
            update={"messages": [AIMessage(content="Passing to writer agent.")]},
            goto="agent_b",
        )
    response = llm.invoke(state["messages"])
    return Command(update={"messages": [response]}, goto=END)
```

---

# 20 — LangGraph Streaming

## `stream_mode` options

| Mode | What it emits | When to use |
|---|---|---|
| `"values"` | Full state dict after each super-step | Debugging, simple monitoring |
| `"updates"` | Per-node state deltas `{node_name: delta}` | Lightweight monitoring |
| `"messages"` | LLM token stream `(AIMessageChunk, metadata)` | UI token streaming |
| `"custom"` | Data from `get_stream_writer()` inside nodes | Bespoke progress events |
| `"checkpoints"` | Checkpoint events | Persistence monitoring |
| `"tasks"` | Task execution events | Fine-grained debugging |
| `"debug"` | Full debug events | Development/tracing |

Multiple modes at once: `stream_mode=["messages", "updates"]`.

**`version="v2"` (recommended):** unified `StreamPart` dict `{"type": ..., "ns": ..., "data": ...}` regardless of mode count or subgraph nesting.

```python
# Unified v2 streaming:
for chunk in graph.stream(input, stream_mode=["messages", "updates"], version="v2"):
    if chunk["type"] == "messages":
        msg, metadata = chunk["data"]
        if hasattr(msg, "content") and msg.content:
            print(msg.content, end="", flush=True)
    elif chunk["type"] == "updates":
        for node, delta in chunk["data"].items():
            print(f"\n[{node}] updated: {list(delta.keys())}")
    elif chunk["type"] == "custom":
        print(f"\n[custom] {chunk['data']}")
```

## `get_stream_writer` — emit custom events from nodes

```python
from langgraph.config import get_stream_writer

def slow_node(state: State) -> dict:
    writer = get_stream_writer()
    writer({"step": "loading", "progress": 0})
    data = load_data()
    writer({"step": "processing", "progress": 50})
    result = process(data)
    writer({"step": "done", "progress": 100})
    return {"result": result}
```

Consumer sees `{"type": "custom", "data": {"step": "loading", "progress": 0}}` etc.

## Token streaming (per-node)

Within a node, you can stream model tokens using `astream_events` on the LLM call:

```python
async def streaming_node(state: State) -> dict:
    full_response = ""
    async for event in llm.astream_events(state["messages"], version="v2"):
        if event["event"] == "on_chat_model_stream":
            token = event["data"]["chunk"].content
            full_response += token
    return {"messages": [AIMessage(content=full_response)]}
```

But for most cases, using `graph.stream(stream_mode="messages")` from the outside is simpler.

## Async streaming

```python
async for chunk in graph.astream(input, stream_mode="messages"):
    msg, meta = chunk
    if msg.content: print(msg.content, end="")
```

---

# 21 — LangGraph Advanced Patterns

## Plan-and-execute

```python
class PlanExecuteState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    plan: list[str]                          # steps from planner
    past_steps: Annotated[list[str], add]   # completed steps
    response: str

def planner(state: PlanExecuteState) -> dict:
    plan_schema = llm.with_structured_output({"type": "object", "properties": {
        "steps": {"type": "array", "items": {"type": "string"}}
    }})
    result = plan_schema.invoke(state["messages"])
    return {"plan": result["steps"]}

def executor(state: PlanExecuteState) -> dict:
    step = state["plan"][0]
    result = agent.invoke({"messages": [HumanMessage(content=step)]})
    return {
        "past_steps": [f"{step}: {result['messages'][-1].content}"],
        "plan": state["plan"][1:],          # consume step
    }

def router(state: PlanExecuteState) -> str:
    if not state["plan"]:
        return "finalize"
    return "execute"

graph = (
    StateGraph(PlanExecuteState)
    .add_node("plan", planner)
    .add_node("execute", executor)
    .add_node("finalize", lambda s: {"response": "\n".join(s["past_steps"])})
    .add_edge(START, "plan")
    .add_edge("plan", "execute")
    .add_conditional_edges("execute", router, {"execute": "execute", "finalize": "finalize"})
    .add_edge("finalize", END)
    .compile()
)
```

## Reflection / self-critique loop

```python
class ReflectionState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    iterations: Annotated[int, lambda a, b: a + b]
    critique: str

def generate(state: ReflectionState) -> dict:
    return {"messages": [llm.invoke(state["messages"])], "iterations": 1}

def critique_and_route(state: ReflectionState) -> str:
    if state["iterations"] >= 3:
        return END
    critique = llm.with_structured_output({"type": "object", "properties": {
        "needs_revision": {"type": "boolean"},
        "feedback": {"type": "string"}
    }}).invoke([
        SystemMessage(content="Is this response high quality? Be strict."),
        state["messages"][-1],
    ])
    if critique["needs_revision"]:
        return "revise"
    return END

def revise(state: ReflectionState) -> dict:
    revision_prompt = [
        SystemMessage(content=f"Revise based on: {state['critique']}"),
        *state["messages"],
    ]
    return {"messages": [llm.invoke(revision_prompt)]}

graph = (
    StateGraph(ReflectionState)
    .add_node("generate", generate)
    .add_node("revise", revise)
    .add_edge(START, "generate")
    .add_conditional_edges("generate", critique_and_route,
                           {"revise": "revise", END: END})
    .add_edge("revise", "generate")
    .compile()
)
```

## Tree-of-Thoughts (via Send)

```python
class ToTState(TypedDict):
    problem: str
    candidates: Annotated[list[str], add]
    scores: Annotated[list[float], add]
    best_solution: str

def generate_candidates(state: ToTState) -> list[Send]:
    return [Send("evaluate_candidate", {"problem": state["problem"], "candidate_id": i})
            for i in range(5)]   # generate 5 candidates in parallel

class CandidateState(TypedDict):
    problem: str
    candidate_id: int

def evaluate_candidate(state: CandidateState) -> dict:
    candidate = llm.invoke(f"Solve: {state['problem']}").content
    score_result = llm.with_structured_output({"type": "object", "properties": {
        "score": {"type": "number"}
    }}).invoke(f"Rate this solution 0-1: {candidate}")
    return {"candidates": [candidate], "scores": [score_result["score"]]}

def select_best(state: ToTState) -> dict:
    best_idx = state["scores"].index(max(state["scores"]))
    return {"best_solution": state["candidates"][best_idx]}

graph = (
    StateGraph(ToTState)
    .add_node("evaluate_candidate", evaluate_candidate)
    .add_node("select", select_best)
    .add_conditional_edges(START, generate_candidates, ["evaluate_candidate"])
    .add_edge("evaluate_candidate", "select")
    .add_edge("select", END)
    .compile()
)
```

## Dynamic graph construction

Build the graph structure at runtime based on configuration:

```python
def build_pipeline(stages: list[str]) -> CompiledStateGraph:
    builder = StateGraph(State)
    prev = START
    for stage in stages:
        node_fn = stage_registry[stage]   # look up function by name
        builder.add_node(stage, node_fn)
        builder.add_edge(prev, stage)
        prev = stage
    builder.add_edge(prev, END)
    return builder.compile()

pipeline = build_pipeline(["load", "transform", "validate", "respond"])
```
