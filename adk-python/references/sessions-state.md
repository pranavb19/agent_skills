# Sessions, state & database connectivity

## Table of contents
1. Concepts: Session, SessionService, Events
2. Database connectivity (deep dive) — drivers, gotchas, full lifecycle
3. State scopes via prefixes
4. Updating state correctly (and the #1 anti-pattern)
5. Instruction templating
6. Events & EventActions

---

## 1. Concepts

A **Session** (`google.adk.sessions.Session`) is one conversation thread, identified by
`(app_name, user_id, id)`, holding `events` (chronological history), `state` (a dict
scratchpad), and `last_update_time`.

**SessionService implementations:**
- `InMemorySessionService` — RAM, lost on restart. Dev/test only.
- `DatabaseSessionService(db_url=...)` — SQLite/Postgres/MySQL via SQLAlchemy; persistent,
  self-managed; **auto-creates its tables on first init**.
- `VertexAiSessionService(project, location, agent_engine_id)` — managed, production on
  GCP / Agent Engine.
- (2.0+) Firestore-backed session service for GCP production.

---

## 2. Database connectivity (deep dive)

The #1 production failure is the **async-driver requirement**: the connection string MUST
use an async dialect or it breaks at session creation.

```python
from google.adk.sessions import DatabaseSessionService

# SQLite (dev / single-user) — note +aiosqlite:
svc = DatabaseSessionService(db_url="sqlite+aiosqlite:///./agent_data.db")

# PostgreSQL (production) — pip install asyncpg
svc = DatabaseSessionService(db_url="postgresql+asyncpg://user:pass@host:5432/agent_sessions")

# MySQL (production) — pip install aiomysql
svc = DatabaseSessionService(db_url="mysql+aiomysql://user:pass@host:3306/agent_sessions")
```

### Mandatory driver matrix (sync dialect silently breaks at runtime)

| DB | ❌ Wrong (sync) | ✅ Correct (async) | pip install |
|---|---|---|---|
| SQLite | `sqlite:///x.db` | `sqlite+aiosqlite:///x.db` | `aiosqlite` |
| Postgres | `postgresql://...` | `postgresql+asyncpg://...` | `asyncpg` |
| MySQL | `mysql://...` | `mysql+aiomysql://...` | `aiomysql` |

### Known gotchas to plan for
- **Schema migration:** the session DB schema changed in ADK Python **v1.22.0** — existing
  DBs need migration (use Alembic). Pin your ADK version alongside the DB and migrate on upgrade.
- **Postgres + asyncpg timezone bug (GitHub #4366):** ADK passes timezone-aware datetimes
  while Postgres `TIMESTAMP` defaults to `WITHOUT TIME ZONE`, raising
  `can't subtract offset-naive and offset-aware datetimes`. Mitigation: use `TIMESTAMPTZ`
  columns / set DB timezone handling, or pin to a patched ADK version.
- **Connection-pool leaks under error load (GitHub #3328):** on commit failures the
  transaction may not roll back, exhausting the pool. Tune via SQLAlchemy URL query params:
  `?pool_size=20&max_overflow=10&pool_recycle=1800&pool_pre_ping=true`.
- **Concurrency:** `DatabaseSessionService` uses two-tier locking (in-process lock to
  serialize `append_event` + DB integrity), but SQLite locks poorly under concurrent
  writes — use Postgres/MySQL for any multi-user deployment.

### Full persistent lifecycle (create → resume → inspect → cleanup)

```python
import asyncio
from google.adk.sessions import DatabaseSessionService
from google.adk.runners import Runner
from google.genai import types

APP, USER = "support_app", "user_42"
svc = DatabaseSessionService(db_url="postgresql+asyncpg://u:p@h:5432/db")

async def main():
    # Resume the user's most recent session, or create a fresh one.
    existing = await svc.list_sessions(app_name=APP, user_id=USER)
    if existing.sessions:
        session = existing.sessions[-1]          # resume — history + state intact
    else:
        session = await svc.create_session(
            app_name=APP, user_id=USER,
            state={"user:tier": "free", "app:promo": "WELCOME10"},  # seed scoped state
        )

    runner = Runner(agent=root_agent, app_name=APP, session_service=svc)
    msg = types.Content(role="user", parts=[types.Part(text="What's my tier?")])
    async for ev in runner.run_async(user_id=USER, session_id=session.id, new_message=msg):
        if ev.is_final_response():
            print(ev.content.parts[0].text)

    reloaded = await svc.get_session(app_name=APP, user_id=USER, session_id=session.id)
    print("Persisted state:", reloaded.state)

    # Housekeeping (e.g. a cron deleting stale threads):
    # await svc.delete_session(app_name=APP, user_id=USER, session_id=session.id)

asyncio.run(main())
```

> All persistent-service methods (`create_session`, `get_session`, `list_sessions`,
> `delete_session`, `append_event`) are **async** — always `await`. Prefer
> **resume-by-listing** (or store `session.id` in your own user table) over hardcoded IDs.

---

## 3. State scopes via key prefixes (critical & frequently missed)

| Prefix | Example | Scope | Persistence |
|---|---|---|---|
| *(none)* | `current_step` | This session only | Persists only with a persistent SessionService |
| `user:` | `user:theme` | All sessions for that `user_id` (within app) | Persisted with DB/Vertex |
| `app:` | `app:discount_code` | All users + sessions for the app | Persisted with DB/Vertex |
| `temp:` | `temp:raw_api` | Current invocation only | **Never** persisted; discarded after invocation |

Use `temp:` as the natural intra-invocation channel between workflow-agent steps.

---

## 4. Updating state correctly (and the #1 anti-pattern)

- ✅ `output_key="my_key"` on an LlmAgent auto-saves the agent's final text/structured
  output to state (the Runner creates the `state_delta` and `append_event`).
- ✅ In tools/callbacks: `tool_context.state["k"] = v` / `callback_context.state["k"] = v`
  — automatically tracked into `EventActions.state_delta`.
- ✅ Outside the runner: build an Event with a delta and append it:
  ```python
  from google.adk.events import Event, EventActions
  evt = Event(author="system", actions=EventActions(state_delta={"user:tier": "pro"}))
  await session_service.append_event(session, evt)
  ```
- ❌ **Never** do `retrieved_session.state["k"] = v` directly — it bypasses event tracking,
  may not persist, and isn't thread-safe.

Best practices: store small, serializable primitives; keep structures shallow; put large
blobs in artifacts (`memory-artifacts.md`), not state.

---

## 5. Instruction templating

`{key}` in an instruction is replaced by `session.state["key"]` before the prompt is sent.
Use `{key?}` for an optional key.

```python
agent = Agent(model="gemini-2.5-flash", name="writer",
              instruction="Write a short post about: {topic}. Audience: {audience?}")
# requires state["topic"]; audience optional
```

---

## 6. Events & EventActions

Events (`google.adk.events.Event`) are the universal message format. Key fields:
`author`, `content`, `actions` (an `EventActions`), `invocation_id`, `timestamp`.

`EventActions` carries: `state_delta`, `artifact_delta`, `transfer_to_agent`, `escalate`,
`skip_summarization`.

`event.is_final_response()` detects the user-facing answer.

Flow: source yields Event → Runner merges `state_delta` into state and appends to history →
Runner yields the Event to the caller.

**2.0 change:** the Event schema gained `node_info` and `output` fields (graph state). Custom
session DBs with rigid columns must be migrated; strict-JSON downstream validators with
`additionalProperties:false` will reject 2.0 events.
