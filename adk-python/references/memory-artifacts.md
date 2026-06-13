# Memory & artifacts

## Part A — Memory (cross-session knowledge)

`MemoryService` provides long-term, cross-session knowledge (vs Session = a single thread).
Use it so the agent "remembers" facts across conversations.

Implementations:
- `InMemoryMemoryService` — keyword search; dev only (naive match).
- `VertexAiMemoryBankService` — Vertex AI Memory Bank; uses Gemini to extract key memories,
  semantic/vector retrieval; production.

### Wiring + populate + retrieve

```python
from google.adk.memory import InMemoryMemoryService
from google.adk.tools import load_memory          # built-in tool: lets the LLM query memory
from google.adk.agents.callback_context import CallbackContext
from google.adk.runners import Runner

memory_service = InMemoryMemoryService()

# Give the agent the ability to recall:
agent = Agent(
    model="gemini-2.5-flash", name="assistant",
    description="Helpful assistant that remembers facts about the user.",
    instruction="Use load_memory to recall facts about the user when relevant.",
    tools=[load_memory],
)

# Persist a finished conversation into long-term memory (commonly in after_agent_callback):
async def ingest_to_memory(callback_context: CallbackContext):
    await memory_service.add_session_to_memory(callback_context._invocation_context.session)
    return None
agent.after_agent_callback = ingest_to_memory

runner = Runner(agent=agent, app_name="app",
                session_service=session_service, memory_service=memory_service)
```

Notes:
- `add_session_to_memory(session)` ingests a session; the agent retrieves via the
  `load_memory` tool (or `search_memory`), which the Runner orchestrates against the
  configured `memory_service`.
- **Memory ≠ state.** Use **state** for the current thread's working data (`sessions-state.md`);
  use **memory** for durable, searchable knowledge across threads.
- With `VertexAiMemoryBankService`, extraction/consolidation is automatic and semantic; the
  in-memory service is naive keyword match (fine for dev).

---

## Part B — Artifacts (versioned binary files)

Artifacts are versioned binary files (images, PDFs, audio) tied to a session or user.

Services (both subclass `BaseArtifactService`): `InMemoryArtifactService` (dev) and
`GcsArtifactService(bucket_name=...)` (production). Provide to the Runner via
`artifact_service=`. Calling `save_artifact` without one raises `ValueError`.

### Save / load via context

```python
from google.genai import types

def save_report(content: str, tool_context) -> dict:
    """Saves a text report as a versioned artifact.

    Args:
        content: the report text.
    Returns:
        dict with status and the new version number.
    """
    part = types.Part.from_bytes(data=content.encode(), mime_type="text/plain")
    version = tool_context.save_artifact(filename="report.txt", artifact=part)
    return {"status": "success", "version": version}
```

### Rules & nuances
- Each save with the same filename creates a **new version** (`save_artifact` returns the int
  version); `list_versions` enumerates them.
- Use the `user:` filename prefix (`"user:profile.png"`) to scope across all of a user's
  sessions; otherwise it's session-scoped.
- **Always set an accurate `mime_type`.**
- `LoadArtifactsTool` lets the LLM discover/request artifact contents on demand.
- For `adk web`, GCS requires `adk web --artifact_service_uri="gs://bucket"`.
- See `inputs.md` for the artifact pattern that routes user-uploaded binary into tools.

Methods on `BaseArtifactService`: `save_artifact`, `load_artifact`, `list_artifact_keys`,
`delete_artifact`, `list_versions`.
