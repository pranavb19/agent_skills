# Overview: install, project layout, CLI, the big picture

## Mental model

An **Agent** (reasoning unit) is run by a **Runner**, which wires together:
- **SessionService** — conversation history + state (the scratchpad).
- **ArtifactService** — binary files (images, PDFs, audio), versioned.
- **MemoryService** — long-term, cross-session knowledge.

The Runner drives an event loop and emits a stream of **Events**. Everything else — tools,
callbacks, sub-agents, A2A — hangs off this spine. Build agents like software: code-first,
model-agnostic, deploy-anywhere.

## Versions & install

- Latest stable: `google-adk` **2.2.0** (PyPI, June 4 2026). ADK **2.0** GA was May 19 2026
  (graph Workflows, collaboration/Task API, breaking changes to agent/event/session schemas).
- Requires Python ≥3.10; **ADK 2.0's new features require Python 3.11+**.
- Install: `pip install google-adk`. Pin 1.x with `pip install "google-adk~=1.0"`.
- Extras: `pip install google-adk[a2a,db,eval,gcp,mcp,extensions]` (also `all`, `community`,
  `tools`, `otel-gcp`, `slack`, `dev`, `test`, `toolbox`, `agent-identity`).
- DB drivers are separate installs (see `sessions-state.md`): `asyncpg`, `aiomysql`, `aiosqlite`.

## Strict project structure (names matter)

```
my_agent/
  __init__.py      # MUST contain: from . import agent
  agent.py         # MUST define module-level: root_agent = ...
  .env             # API keys / project IDs
```

`__init__.py`:
```python
from . import agent
```

`.env` (AI Studio key) or Vertex:
```
GOOGLE_API_KEY=...                 # AI Studio
# or, for Vertex AI:
GOOGLE_GENAI_USE_VERTEXAI=TRUE
GOOGLE_CLOUD_PROJECT=my-project
GOOGLE_CLOUD_LOCATION=us-central1
```

## Minimal agent (mandatory docstring discipline)

```python
from google.adk.agents import Agent  # Agent is an alias for LlmAgent

def get_current_time(city: str) -> dict:
    """Returns the current time in a specified city.

    Use this whenever the user asks what time it is somewhere.

    Args:
        city: The city name to look up, e.g. "Tokyo".
    Returns:
        dict with keys: status ("success"|"error"), city, time.
    """
    return {"status": "success", "city": city, "time": "10:30 AM"}

root_agent = Agent(
    model="gemini-2.5-flash",
    name="root_agent",
    description="Answers general questions and reports the time in cities.",  # routing signal
    instruction="Answer questions; call get_current_time for time queries.",
    tools=[get_current_time],
)
```

## CLI

| Command | Purpose |
|---|---|
| `adk create my_agent` | Scaffold a new agent project |
| `adk run my_agent` | Interactive CLI chat |
| `adk web` | Dev UI on :8000 (Trace/Events/Eval tabs). **Dev only — never production** |
| `adk api_server` | FastAPI server exposing `/run`, `/run_sse`, session endpoints |
| `adk eval` | Run evaluations against EvalSets |
| `adk deploy agent_engine\|cloud_run\|gke` | Deploy to a backend (see `deployment.md`) |

The dev UI is for development/debugging only; use a real server + persistent services in prod.

## Where to go next

- Run loop & config → `runner-runtime.md`
- Persistence & state → `sessions-state.md`
- Agent types & delegation → `agents.md`
- Tools → `tools.md`
- Inputs/outputs → `inputs.md`, `outputs.md`
