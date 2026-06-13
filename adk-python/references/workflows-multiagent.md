# Multi-agent design patterns & the 2.0 graph Workflow API

## Part A — Classic multi-agent patterns

- **Coordinator/Dispatcher:** a root LlmAgent routes to specialists via LLM-driven delegation
  (`sub_agents` with good `description`s) or explicit `AgentTool` invocation. Use AgentTool
  when the coordinator must retain control and synthesize results.
- **Sequential pipeline:** `SequentialAgent` with `output_key` passing data step→step
  (parse → extract → summarize).
- **Parallel fan-out / gather:** `ParallelAgent` for independent subtasks, then a
  `SequentialAgent` whose final agent reads each branch's `output_key` and synthesizes.
- **Generator–Critic loop:** `LoopAgent` wrapping generator + critic; critic emits `PASS` or
  calls an exit/escalate tool to break.
- **Hierarchical decomposition:** multi-level trees where parents delegate and results bubble
  up via tool responses or shared state.
- **Shared-state communication:** the passive default — one agent writes `state['x']`,
  another reads it (works because workflow agents share the InvocationContext). `temp:` is the
  natural intra-invocation channel.

### When workflow agents vs LLM-driven delegation
- Use **workflow agents** when order/structure is a business requirement — deterministic,
  testable, cheaper, no extra LLM routing calls.
- Use **LLM-driven delegation** when routing must adapt to content.

### Pitfalls
Monolithic do-everything agents; over-deep hierarchies (delegation degrades after the first
transfer); passing full history to a simple function (use AgentTool/tool instead);
built-in-tool-in-sub-agent violations (`tools.md`).

### Coordinator example
```python
from google.adk.tools.agent_tool import AgentTool

billing = Agent(model="gemini-2.5-flash", name="billing",
                description="Handles invoices, refunds, payment questions.")
tech    = Agent(model="gemini-2.5-flash", name="tech",
                description="Handles bugs, errors, technical troubleshooting.")

root = Agent(
    model="gemini-2.5-flash", name="support_coordinator",
    description="Front-line support router.",
    instruction=("Route billing questions to the billing tool and technical questions to the "
                 "tech tool. Never answer billing or technical questions yourself."),
    tools=[AgentTool(agent=billing), AgentTool(agent=tech)],
)
```

---

## Part B — ADK 2.0 graph Workflow API (badged v2.0.0)

Adds deterministic DAG execution, collaboration modes, and human-in-the-loop. Verify exact
constructor signatures against your installed version — docs show usage by example.

### Building a DAG with `edges`
```python
from google.adk import Agent, Workflow, Event

def router(node_input: str):
    if "urgent" in node_input.lower():
        return Event(route="PRIORITY")
    return Event(route="STANDARD")

root_agent = Workflow(
    name="support_router",
    edges=[
        ("START", classifier_agent, router),                       # unconditional, in order
        (router, {"PRIORITY": priority_handler, "STANDARD": standard_handler}),  # conditional
    ],
)
```

- **Unconditional edges:** a tuple starting with `"START"` listing nodes in order.
- **Parallel:** repeat `"START"` in multiple tuples.
- **Conditional:** `(router_node, {route_value: target_node})` where the router returns
  `Event(route=...)`.
- **Fan-in:** `JoinNode` (`from google.adk.workflow import JoinNode`) — proceeds only after
  ALL upstream nodes emit.

### FunctionNode
A plain Python function passed into edges, receiving `node_input` and returning
`Event(output=...)`; the next node receives that payload as its `node_input`. `Event` also
carries `message` (to user), `state` (persisted), and `route`.

### Data handling between nodes
Nodes accept `input_schema` / `output_schema` (Pydantic `BaseModel`); reference fields in
instructions with `{Schema.field}` or `<Schema.field from source_node>`.

### Dynamic workflows
`@node` decorator (`from google.adk.workflow import node`), `BaseNode`, `ctx.run_node(...)`,
and `rerun_on_resume=True`.

### Collaboration modes (`mode=` on subagent Agents only — never on the root)
- `"chat"` (default) — full user interaction; manual return to parent.
- `"task"` — clarifications allowed; auto-returns to parent on completion
  (`complete_task`); must be a leaf agent. **Disabled inside graph workflows in v2.0.0.**
- `"single_turn"` — no user interaction; auto-returns; parallel-capable.

### Human-in-the-loop
```python
from google.adk.events import RequestInput
# inside a node:
yield RequestInput(message="Approve this action?", response_schema=UserFeedback)  # pauses execution
```

> Version note: docs badge these features "v2.0.0" while PyPI's latest stable is 2.2.0.
> Treat the surface above as the 2.0 GA surface; verify exact signatures against the installed
> version's API reference.
