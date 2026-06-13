---
name: adk-python
description: >-
  Build, debug, and deploy agentic and multi-agent systems with Google's Agent
  Development Kit (ADK) for Python (google-adk 2.x). Use this skill whenever the
  user mentions ADK, "Agent Development Kit", google-adk, LlmAgent, SequentialAgent /
  ParallelAgent / LoopAgent, sub_agents, AgentTool, the A2A protocol, RemoteA2aAgent,
  Runner / SessionService / ArtifactService / MemoryService, output_schema / output_key,
  adk web / adk run / adk deploy, or asks to wire up tools, callbacks, multimodal/file
  inputs, structured outputs, persistent (database) sessions, memory, workflow graphs,
  or deployment (Vertex AI Agent Engine, Cloud Run, GKE) for a Python agent. Use it even
  when the user only describes the behavior ("an agent that calls tools and remembers
  users across chats", "route a request to specialist agents", "make my agent return
  validated JSON", "persist conversations in Postgres") without naming ADK explicitly.
  It encodes the correct, current APIs and the mandatory syntax (tool docstrings, agent
  descriptions, async DB drivers, callback parameter names, output_schema constraints)
  that are easy to get wrong from memory.
---

# Google ADK (Python) — Agent & Multi-Agent Development

Authoritative, code-first reference for building production agentic systems with
`google-adk` 2.x. Training-data memory of ADK is frequently stale or wrong — **consult
the reference files below instead of recalling APIs**, especially for tool signatures,
database URLs, `output_schema` rules, callback parameter names, and the 2.0 graph API.

## How to use this skill

1. Identify which area(s) the task touches (see the routing table).
2. **Open the matching `references/*.md` file(s) and follow them** before writing code.
   Several files often apply at once (e.g. a multi-agent system that takes file input and
   persists sessions = `agents.md` + `inputs.md` + `sessions-state.md`).
3. Honor the **Mandatory syntax** rules below in every snippet — these are the things that
   silently misbehave or hard-error when gotten wrong.
4. When unsure of the installed version's exact signatures, say so and verify; ADK ships
   roughly bi-weekly and 2.0 introduced breaking changes.

## Mandatory syntax (never get these wrong)

1. **Tool docstrings are the API contract.** The LLM sees only the function name, type
   hints, and docstring. Document every parameter (what it is / when to use it / what it
   returns). **Never document `tool_context`** — it is framework-injected; mentioning it
   confuses the model.
2. **Tool parameters: no default values; JSON-serializable types only.** Return a `dict`
   (ideally with a `status` key like `"success"`/`"error"`). You cannot pass raw bytes
   through tool params — use the artifact pattern (`references/inputs.md`).
3. **Agent `name` + `description` are routing identifiers.** In multi-agent systems the
   parent LLM picks children by their `description`; vague descriptions cause wrong
   routing. `name` must be a valid identifier, unique among siblings.
4. **Callback parameter names are fixed:** `callback_context` (agent/model callbacks) and
   `tool_context` (tool callbacks), passed by keyword — renaming raises `TypeError`.
5. **`output_schema` disables tools AND transfer** for that agent. On a sub-agent you must
   also set `disallow_transfer_to_parent=True, disallow_transfer_to_peers=True`, and wrap
   runs in try/except for `pydantic.ValidationError`.
6. **Database URLs must use an async dialect:** `sqlite+aiosqlite://`,
   `postgresql+asyncpg://`, `mysql+aiomysql://`. Sync dialects break at runtime.
7. **Project files are name-bound:** `__init__.py` must contain `from . import agent`;
   `agent.py` must expose a module-level variable named `root_agent`.
8. **Never mutate `session.state` directly.** Write via `output_key`,
   `*_context.state[...] = ...`, or `EventActions(state_delta=...)` + `append_event`.

## Routing table — which reference to read

| Task / trigger | Read |
|---|---|
| Install, project layout, CLI, first agent, the big picture | `references/overview.md` |
| Runner, RunConfig, invocation loop, `run_async`/`run`/`run_live`, `max_llm_calls` | `references/runner-runtime.md` |
| Sessions, **database connectivity** (SQLite/Postgres/MySQL), state scopes/prefixes, Events | `references/sessions-state.md` |
| Agent types (LlmAgent, Sequential/Parallel/Loop, custom), sub_agents vs AgentTool, transfer/delegation | `references/agents.md` |
| Tools: function tools, ToolContext, built-in tools + the one-per-agent limit, **conflicting/ambiguous tools**, OpenAPI/MCP/3p, auth | `references/tools.md` |
| **Inputs**: text + multiple files, multimodal (image/PDF/audio), binary-via-artifact, `input_schema`, passing data between agents | `references/inputs.md` |
| **Outputs**: `output_schema` + `output_key`, validated JSON, structure + tools coexistence | `references/outputs.md` |
| Configuration: model, `generate_content_config`, instructions, planners, code executors, **callbacks** (6 hooks), Plugins | `references/configuration.md` |
| **Multi-agent design patterns** + the ADK **2.0 graph Workflow** API, collaboration modes, human-in-the-loop | `references/workflows-multiagent.md` |
| **A2A protocol**: `to_a2a`, `RemoteA2aAgent`, agent cards, security | `references/a2a.md` |
| **Memory** (cross-session), `MemoryService`, `load_memory`; Artifacts & `ArtifactService` | `references/memory-artifacts.md` |
| Models: Gemini, LiteLLM, routing/fallback; streaming (SSE + bidirectional live audio/video) | `references/models-streaming.md` |
| Deployment: Agent Engine, Cloud Run, GKE, production sessions, scaling | `references/deployment.md` |
| Evaluation (`adk eval`), tracing/observability (OpenTelemetry/Cloud Trace), debugging, safety/guardrails, cost/perf, 2.0 migration gotchas | `references/eval-observability-ops.md` |

## Default workflow for "build me an ADK agent/system"

1. **Scaffold** the name-bound structure (`references/overview.md`).
2. **Single agent first** — clean function tools with full docstrings, dict returns
   (`references/tools.md`). Verify tool selection before adding complexity.
3. **Inputs/outputs** — multimodal via multi-part `Content`, binary via the artifact
   pattern; machine-consumed output via `output_schema`+`output_key`
   (`references/inputs.md`, `references/outputs.md`).
4. **Decompose** into specialists only when warranted; choose Sequential/Parallel/Loop for
   known control flow, coordinator+AgentTool when the root must keep control, `sub_agents`
   transfer for genuinely stateful handoffs (`references/agents.md`,
   `references/workflows-multiagent.md`).
5. **Productionize** — persistent `DatabaseSessionService` (async driver), GCS artifacts,
   `MemoryService`, deployment target, tracing, guardrails, `adk eval`
   (`references/sessions-state.md`, `references/memory-artifacts.md`,
   `references/deployment.md`, `references/eval-observability-ops.md`).

## Version note

Latest stable is `google-adk` 2.2.0 (June 2026); ADK 2.0 GA (May 2026) added graph
Workflows, a collaboration/Task API, and breaking changes to the agent/event/session
schemas (Python 3.11+ for 2.0 features). The reference files flag version-dependent
behavior and known GitHub issues. When prose docs and PyPI disagree on exact constructor
signatures, prefer the installed version's API reference and verify.
