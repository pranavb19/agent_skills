# Runner & runtime

## What the Runner is

The **Runner** is the orchestrator. Responsibilities: load/resume the Session via the
SessionService, build the InvocationContext, drive the agent's event loop, route Events,
persist state/artifact deltas through `append_event`, and yield Events to the caller.

It is **stateless and thread-safe** — a single Runner can serve thousands of concurrent
users because all per-user state lives in the services. Scale horizontally on a shared
persistent SessionService.

## Constructing and running

```python
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

session_service = InMemorySessionService()
runner = Runner(agent=root_agent, app_name="my_app", session_service=session_service)

async def call(query, user_id, session_id):
    content = types.Content(role="user", parts=[types.Part(text=query)])
    async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=content):
        if event.is_final_response():
            print(event.content.parts[0].text)
```

- `runner.run_async(...)` — async generator; the standard production path.
- `runner.run(...)` — synchronous wrapper (simple scripts / CLI).
- `runner.run_live(...)` — bidirectional live streaming (audio/video); see `models-streaming.md`.
- `InMemoryRunner(agent=...)` — bundles in-memory session/artifact/memory services for quick starts.

Constructor args: `agent`, `app_name`, `session_service`, and optional `artifact_service`,
`memory_service`, and (2.0) a credential service. **Calling `save_artifact` without passing an
`artifact_service` raises `ValueError`.**

## RunConfig — per-run behavior

Passed to `run_async`/`run_live`:

```python
from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.runners import GetSessionConfig  # location may vary by version

cfg = RunConfig(
    streaming_mode=StreamingMode.SSE,      # NONE (default) | SSE | BIDI
    max_llm_calls=200,                     # default 500; 0/negative = unlimited
    get_session_config=GetSessionConfig(num_recent_events=50),  # cost/latency lever
)
async for ev in runner.run_async(user_id=u, session_id=s, new_message=msg, run_config=cfg):
    ...
```

Key fields:
- `streaming_mode`: `StreamingMode.NONE` (one response/turn), `SSE` (token streaming),
  `BIDI` (WebSocket live).
- `max_llm_calls` (default **500**): caps total LLM calls per run to prevent runaway
  loops/costs. **0 or negative = unlimited**; a value at/above `sys.maxsize` raises
  `ValueError`. **Does NOT apply to `run_live()` BIDI** — implement your own guardrails there.
- `support_cfc=True`: compositional function calling (experimental, Gemini 2.x).
- `get_session_config=GetSessionConfig(num_recent_events=N)`: limits the history loaded per
  invocation — a major cost/latency optimization on long sessions.
- Live-only: `response_modalities`, `speech_config`, `output_audio_transcription`,
  `session_resumption`, `context_window_compression`, `save_live_blob=True`
  (replaces deprecated `save_live_audio`).
- `custom_metadata`: arbitrary tags attached to events.

## The invocation loop (what happens per turn)

1. Runner gets/creates the Session from the SessionService.
2. Builds an InvocationContext (carries session, services, invocation_id, and `temp:` state).
3. Calls the root agent, which yields Events (model output, tool calls/results, transfers).
4. For each Event, the Runner merges `actions.state_delta` into state and appends the Event
   to history via `append_event`, then yields it to the caller.
5. Loop continues until the agent produces a final response (`event.is_final_response()`).

## Common mistakes

- Forgetting `await` on async service methods.
- Hardcoding a `session_id` instead of resuming the user's latest (see `sessions-state.md`).
- Relying on `max_llm_calls` to bound a `run_live` BIDI session (it won't).
- Doing heavy blocking work inside the loop (callbacks) on the critical path.
