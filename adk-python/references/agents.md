# Agent types & delegation

## Agent types

### LlmAgent / Agent
LLM-driven reasoning, tool use, delegation. `name` and `description` are **required** and
act as the service-discovery interface other agents use to route to it. `Agent` is an alias
for `LlmAgent`.

### Workflow (template) agents — deterministic orchestration of `sub_agents`
- **`SequentialAgent(sub_agents=[...])`** — runs children in strict order; passes the SAME
  InvocationContext, so `temp:` state and `output_key` values flow between steps.
- **`ParallelAgent(sub_agents=[...])`** — runs children concurrently; ideal for independent
  fan-out. Each branch should write to a distinct `output_key`; a downstream agent reads them.
- **`LoopAgent(sub_agents=[...], max_iterations=N)`** — repeats until `max_iterations` or a
  sub-agent escalates (`tool_context.actions.escalate = True`, often via an `exit_loop` tool).

```python
from google.adk.agents import SequentialAgent, ParallelAgent, LoopAgent

pipeline = SequentialAgent(name="pipeline", sub_agents=[parser, extractor, summarizer])
fanout   = ParallelAgent(name="fanout", sub_agents=[news_agent, weather_agent, stocks_agent])
refine   = LoopAgent(name="refine", max_iterations=5, sub_agents=[generator, critic])
```

### Custom agents — subclass BaseAgent
Implement `_run_async_impl(self, ctx)` as an async generator yielding Events. Construct
internal composite agents in `__init__` **before** `super().__init__()`, list them in
`sub_agents`, and store references as instance attributes (Pydantic validates).

```python
from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from typing import AsyncGenerator

class MyAgent(BaseAgent):
    def __init__(self, name, child):
        super().__init__(name=name, sub_agents=[child])
        self.child = child

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        async for ev in self.child.run_async(ctx):
            yield ev
        # custom branching, retries, etc.
```

**2.0 caveat:** `BaseAgent` now subclasses `BaseNode`; the graph engine bypasses legacy
`_run_async_impl`/`generate_content` overrides for graph execution. Move custom logic into
callbacks or use the new node model (`workflows-multiagent.md`).

## Sub-agent hierarchy & transfer

`sub_agents=[...]` builds a parent/child tree. LLM-driven delegation issues a
`transfer_to_agent` action; **control fully transfers** to the child, and by default
subsequent user input is answered by that child.

- A sub-agent **cannot** transfer back to its parent with `transfer_to_agent` by default.
  To return control, set `tool_context.actions.escalate = True` (via a custom tool), or use
  2.0 collaboration `task` mode (auto-returns).
- `disallow_transfer_to_parent` / `disallow_transfer_to_peers` flags exist; transfer-up is
  intentionally not the default path.
- **Common pitfall (GitHub #2994 / #3878):** after the first transfer, the root may try to
  call sub-agent-only tools directly, or delegation becomes inconsistent. Mitigations:
  strong coordinator instructions ("you MUST call X / transfer; never answer directly"),
  flatter hierarchies, and clear, disjoint `description`s.

## sub_agents vs AgentTool

| | `sub_agents` (transfer) | `AgentTool` (agent-as-tool) |
|---|---|---|
| Control | Fully handed to child | Caller keeps control; gets result back |
| Context | Shares session + history | Encapsulated call with structured I/O |
| Use for | Stateful, multi-step, context-dependent processes | Discrete, stateless, reusable "expert function" |
| Multimodal | Propagates events (incl. images) to next LLM turn | Wraps result as dict; multimodal returns can be lost (GitHub #729) |

```python
from google.adk.tools.agent_tool import AgentTool

root = Agent(
    name="root", model="gemini-2.5-flash",
    description="Coordinator that delegates summarization.",
    tools=[AgentTool(agent=summarizer_agent, skip_summarization=True)],
)
```

- `skip_summarization=True` bypasses the LLM re-summarizing a well-formatted tool result.
- State set by an AgentTool-wrapped agent propagates back to the parent session.

**Choosing wrong is a top source of bugs:** `sub_agents` when you wanted call-and-return
causes context loss / runaway delegation; `AgentTool` when you needed a stateful handoff
loses shared history.

See `workflows-multiagent.md` for full design patterns and the 2.0 graph API.
