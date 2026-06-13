# A2A (Agent-to-Agent) protocol

A2A is an open standard (spec v0.2: stateless interactions + standardized OpenAPI-style auth)
for cross-framework, cross-vendor agent communication via **Agent Cards** (JSON "business
cards" at `/.well-known/agent-card.json` advertising skills/capabilities).

## Exposing an ADK agent over A2A

```python
from google.adk.a2a import to_a2a  # converts agent → ASGI/Starlette app

a2a_app = to_a2a(root_agent)        # auto-generates the agent card
# run with: uvicorn my_module:a2a_app --port 8001
```

`to_a2a()` sets up an `A2aAgentExecutor` (the bridge between A2A and ADK), an
`InMemoryTaskStore`, an `InMemoryPushNotificationConfigStore`, a `DefaultRequestHandler`, and a
Starlette app that serves the card. Alternatively `adk api_server --a2a` or `adk deploy` can
expose agents.

## Consuming a remote agent

```python
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent, AGENT_CARD_WELL_KNOWN_PATH

prime_agent = RemoteA2aAgent(
    name="prime_agent",
    description="Checks if numbers are prime.",
    agent_card=f"http://localhost:8001/a2a/check_prime_agent{AGENT_CARD_WELL_KNOWN_PATH}",
    use_legacy=False,
)
# use as a sub_agent or AgentTool of your root agent
```

- `RemoteA2aAgent` translates between A2A protocol messages and native ADK Events; customize
  via `A2aRemoteAgentConfig` converter hooks (`a2a_message_converter`, `a2a_task_converter`,
  `a2a_status_update_converter`, `a2a_artifact_update_converter`, `a2a_part_converter`).

## Relationship to ADK (when to use A2A vs in-process)

A2A is the transport/protocol; ADK is the framework. Use A2A to cross **process or
organization boundaries**. For in-process composition use `sub_agents` / `AgentTool` — no
network overhead (`agents.md`).

## Security / auth

A2A v0.2 uses OpenAPI-style auth schemes declared in the agent card; on GCP deployments agents
inherit managed auth. When testing locally, the exposed (remote) agent and the consuming agent
must use **different ports** (e.g., 8001 vs 8000).
