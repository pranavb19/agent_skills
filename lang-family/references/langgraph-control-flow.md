# 13 — LangGraph Control Flow: Edges, `Send`, and `Command`

---

## Static edges

```python
builder.add_edge("node_a", "node_b")   # node_a always goes to node_b
builder.add_edge(START, "entry")       # entry point
builder.add_edge("last", END)          # exit point
```

A node can have multiple outgoing static edges (fan-out to multiple nodes in the same step):
```python
builder.add_edge("dispatcher", "worker_a")
builder.add_edge("dispatcher", "worker_b")  # both worker_a and worker_b run in parallel
```

---

## `add_conditional_edges` — dynamic routing

The router function takes state (and optionally config) and returns either:
- A string: the name of the next node (or `END`)
- A list of strings: multiple next nodes (fan-out)

```python
def router(state: State) -> str:
    if state["status"] == "tool_needed":
        return "tools"
    elif state["status"] == "done":
        return END
    else:
        return "agent"

builder.add_conditional_edges(
    "agent",             # source node
    router,              # routing function
    {                    # optional name mapping (useful when router returns keys not node names)
        "tools": "tool_node",
        "agent": "agent",
        END: END,
    }
)
```

**Without the mapping dict:**
```python
# If router returns the exact node name or END, the mapping is optional:
builder.add_conditional_edges("agent", router)
```

**Returning multiple next nodes (parallel fan-out):**
```python
def fanout(state: State) -> list[str]:
    return ["worker_a", "worker_b", "worker_c"]  # all run in parallel

builder.add_conditional_edges(START, fanout, ["worker_a", "worker_b", "worker_c"])
# Third arg is required when returning a list — it lists all possible next nodes.
```

**`tools_condition` — prebuilt conditional for agents:**
```python
from langgraph.prebuilt import tools_condition
# Returns "tools" if last message has tool_calls, else END
builder.add_conditional_edges("agent", tools_condition)
```

---

## `Send` API — map-reduce / dynamic fan-out

`Send` creates a dynamic number of parallel node invocations, each with their own sub-state. Returns are accumulated via reducers.

**Key difference from static fan-out:** `Send` lets each parallel invocation receive *different inputs* and allows the sub-state to be a different shape than the parent state.

```python
from langgraph.types import Send
from typing import Annotated
from operator import add

class ParentState(TypedDict):
    subjects: list[str]                        # input: list of subjects
    jokes: Annotated[list[str], add]           # output: accumulated from workers

class WorkerState(TypedDict):                  # worker has its own state shape
    subject: str

def generate_joke(state: WorkerState) -> dict:
    joke = f"Why did the {state['subject']} cross the road? Because it could!"
    return {"jokes": [joke]}   # returned to ParentState via the 'jokes' reducer

def fan_out(state: ParentState) -> list[Send]:
    return [Send("generate_joke", {"subject": s}) for s in state["subjects"]]

graph = (
    StateGraph(ParentState)
    .add_node("generate_joke", generate_joke)
    .add_conditional_edges(START, fan_out, ["generate_joke"])  # third arg: list of target nodes
    .add_edge("generate_joke", END)
    .compile()
)

result = graph.invoke({"subjects": ["chicken", "robot", "lawyer"], "jokes": []})
print(result["jokes"])   # 3 jokes accumulated in arbitrary order
```

**`Send` semantics:**
- `Send(node_name, state)` — schedules `node_name` with `state` as input.
- `state` is the **node's input**, not necessarily the parent state. The worker node's return dict is merged into the parent state via reducers.
- Workers run in parallel (as a single super-step).
- Order of results is non-deterministic — use a reducer that handles unordered accumulation (`add`, `lambda a, b: sorted(a+b)`, etc.).
- Mix of static edges and `Send` in the same router is allowed: return a list that includes both `Send` objects and string node names.

```python
# Mix of Send and static routes:
def smart_fanout(state: ParentState):
    result = []
    if state["do_parallel"]:
        result.extend([Send("worker", {"item": x}) for x in state["items"]])
    else:
        result.append("sequential_worker")  # string node name
    return result
```

---

## `Command` — combine state update + routing in one return

`Command` lets a node simultaneously update state AND specify the next node — without needing a separate conditional edge. This makes nodes more self-contained.

```python
from langgraph.types import Command
from typing import Literal

# Node that decides its own next step:
def smart_node(state: State) -> Command[Literal["step_a", "step_b", END]]:
    if state["score"] > 0.8:
        return Command(
            update={"result": "high_score", "processed": True},
            goto="step_a",
        )
    elif state["score"] > 0.5:
        return Command(update={"result": "medium"}, goto="step_b")
    else:
        return Command(update={"result": "low_score"}, goto=END)
```

**`Command` fields:**
- `update: dict` — partial state update (same semantics as returning a dict from a node).
- `goto: str | list[str] | Send | list[Send]` — next node(s), or `END`.
- `graph: str | None` — use `Command.PARENT` to hop from a subgraph into its parent.
- `resume: Any` — used when resuming from an interrupt (see `14-langgraph-interrupts-hitl.md`).

**Routing annotation:** annotate the return type `Command[Literal["node1", "node2", END]]` so LangGraph can render the graph statically. This annotation is optional at runtime but strongly recommended.

**`Command` with `Send` in `goto`:**
```python
def batch_dispatcher(state: State) -> Command:
    sends = [Send("processor", {"item": x}) for x in state["items"]]
    return Command(
        update={"dispatched": True},
        goto=sends,           # fan-out via Command
    )
```

**No outgoing edges from `Command` nodes:** when a node returns `Command`, static edges from that node are ignored — `Command.goto` takes over. Don't add `add_edge` from a `Command`-returning node unless you have a fallback (non-Command) return path.

---

## Cross-graph hops with `Command.PARENT`

From inside a subgraph node, hop to a node in the parent graph:

```python
def subgraph_node(state: SubState) -> Command:
    return Command(
        update={"subgraph_result": "done"},
        goto="parent_node",           # node name in the PARENT graph
        graph=Command.PARENT,         # tells LangGraph to look up one level
    )
```

**Requirement:** the key `"subgraph_result"` (or whatever you update) must exist in the parent state schema AND must have a reducer. Without a reducer in the parent, `Command.PARENT` updates will raise `InvalidUpdateError`.

---

## Entry point alternatives

```python
# Using START (standard):
builder.add_edge(START, "entry_node")

# Using set_entry_point (alias):
builder.set_entry_point("entry_node")

# Conditional entry point:
builder.set_conditional_entry_point(router_fn, {"case_a": "node_a", "case_b": "node_b"})
```

---

## Finish point alternatives

```python
# Using END (standard):
builder.add_edge("final_node", END)

# Using set_finish_point (alias):
builder.set_finish_point("final_node")
```

---

## `interrupt_before` / `interrupt_after` at compile time

Static breakpoints — always pause at these nodes (when a checkpointer is set):

```python
compiled = builder.compile(
    checkpointer=ckpt,
    interrupt_before=["tools"],       # pause before executing the tools node
    interrupt_after=["research"],     # pause after the research node completes
)
```

These are static — they apply to every run. For dynamic/conditional interrupts, use `interrupt()` inside the node (see `14-langgraph-interrupts-hitl.md`).

---

## Routing decision guide

| Pattern | Use |
|---|---|
| Always go A → B | `add_edge("A", "B")` |
| Go A → B or C based on state | `add_conditional_edges("A", router)` |
| Fan out to N dynamic workers with different inputs | `Send` from conditional edge |
| Node decides its own next step + updates state | `Command(update=..., goto=...)` |
| Parallel static fan-out | Multiple `add_edge` from same source |
| Jump from subgraph to parent | `Command(goto=..., graph=Command.PARENT)` |
| Pause before every tool call | `compile(interrupt_before=["tools"])` |
| Pause conditionally from inside a node | `interrupt()` (see file 14) |

---

## Gotchas

1. **`Send` result order is non-deterministic.** Workers run in parallel; which one finishes first is undefined. Use reducers that handle unordered input (`operator.add` is fine; sorted or positional operations are not).

2. **`Command.goto` overrides static edges.** If you add an `add_edge` from a node that can return `Command`, that static edge is only used when the node returns a plain dict (no `Command`). When it returns `Command`, `goto` is authoritative.

3. **`Command(graph=Command.PARENT)` requires reducers in parent state.** Any key updated via `Command.PARENT` must have a reducer in the parent schema, or you get `InvalidUpdateError`.

4. **Conditional edge routing functions run on the CPU synchronously.** Don't do expensive I/O inside routing functions — they block the graph execution loop. If you need an LLM to decide routing, that logic should be in a node, and the routing function reads the result from state.

5. **`add_conditional_edges` with a list return requires the third arg.** When the router can return multiple node names, you must pass the complete list of possible destination nodes as the third argument. This is used for static graph analysis and visualization.

6. **`tools_condition` returns `"tools"` or `END`** — it reads `state["messages"][-1].tool_calls`. If the last message is not an `AIMessage` or has no `tool_calls`, it routes to `END`. Make sure the model's output is the last message before calling `tools_condition`.
