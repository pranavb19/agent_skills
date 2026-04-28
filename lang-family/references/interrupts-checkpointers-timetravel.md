# 14 — LangGraph Interrupts and Human-in-the-Loop

---

## `interrupt()` — pause execution from inside a node

`interrupt(value)` pauses the graph at the current node and surfaces `value` to the caller. Execution resumes from the same point when `Command(resume=...)` is provided. **Requires a checkpointer.**

```python
from langgraph.types import Command, interrupt

def human_review_node(state: State) -> dict:
    # Ask the human a question:
    user_response = interrupt({
        "question": "Please review this draft:",
        "draft": state["messages"][-1].content,
    })
    # user_response is whatever was passed to Command(resume=...)
    # Execution continues here after resume.
    if user_response.get("approved"):
        return {"status": "approved"}
    else:
        return {"status": "rejected", "feedback": user_response.get("feedback")}
```

**Callee side:**
```python
from langgraph.checkpoint.memory import InMemorySaver

compiled = builder.compile(checkpointer=InMemorySaver())
cfg = {"configurable": {"thread_id": "review-1"}}

# First invocation — graph runs until interrupt:
out = compiled.invoke(initial_state, cfg)
# out["__interrupt__"] contains [Interrupt(value={...}, resumable=True, ...)]
print(out["__interrupt__"][0].value)  # {"question": "Please review...", "draft": "..."}

# Human reviews, then resume:
result = compiled.invoke(Command(resume={"approved": True}), cfg)
# or:
result = compiled.invoke(Command(resume={"approved": False, "feedback": "Too verbose"}), cfg)
```

---

## Multiple `interrupt()` calls in one node

Each `interrupt()` call in a node is matched **by index** to the resume list. The node re-executes from the top on every resume — earlier `interrupt()` calls return their previously-resumed values immediately (from the checkpoint).

```python
def multi_question_node(state: State) -> dict:
    # First interrupt:
    name = interrupt("What is your name?")
    # On resume #1: name = whatever was passed to Command(resume=...)

    # Second interrupt (only reached after resume #1):
    age = interrupt(f"Hello {name}! What is your age?")
    # On resume #2: age = whatever was passed to Command(resume=...)

    return {"user_name": name, "user_age": age}
```

**Resume sequence:**
```python
# Turn 1: triggers first interrupt
out = compiled.invoke({}, cfg)
# out["__interrupt__"][0].value = "What is your name?"

# Turn 2: resume with name → triggers second interrupt
out = compiled.invoke(Command(resume="Alice"), cfg)
# out["__interrupt__"][0].value = "Hello Alice! What is your age?"

# Turn 3: resume with age → node completes
result = compiled.invoke(Command(resume=30), cfg)
```

**Critical rules:**
- `interrupt()` calls MUST be in deterministic order — never inside a non-deterministic branch (e.g., random or time-based conditionals).
- The node runs from the top on each resume. Any side effects before `interrupt()` will re-run. Make side effects idempotent.
- Don't `interrupt()` inside a `try/except` that might swallow `NodeInterrupt` — let it propagate.

---

## Static breakpoints (`interrupt_before` / `interrupt_after`)

Compile-time breakpoints that pause on every execution at the specified nodes:

```python
compiled = builder.compile(
    checkpointer=ckpt,
    interrupt_before=["tools"],    # pause before tools node runs
    interrupt_after=["research"],  # pause after research node completes
)

cfg = {"configurable": {"thread_id": "t1"}}
out = compiled.invoke(initial, cfg)
# Paused before "tools" node

# Inspect state:
state = compiled.get_state(cfg)
print(state.values)     # full current state
print(state.next)       # ("tools",) — the next node to run

# Approve (continue execution):
result = compiled.invoke(None, cfg)  # pass None to just continue

# Or edit the state before continuing:
compiled.update_state(cfg, {"messages": [edited_message]})
result = compiled.invoke(None, cfg)
```

---

## `HumanInTheLoopMiddleware` (agent-level, simpler)

For agents using `create_agent`, the middleware handles HITL automatically (see `07-middleware.md`):

```python
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware

agent = create_agent(
    model=llm, tools=tools,
    middleware=[HumanInTheLoopMiddleware(tools_requiring_approval=["delete"])],
    checkpointer=ckpt,
)
# Automatically interrupts before dangerous tools and provides resume protocol.
```

---

## Interrupt object structure

```python
from langgraph.types import Interrupt

# state["__interrupt__"] is list[Interrupt]
intr: Interrupt = state["__interrupt__"][0]
intr.value          # whatever was passed to interrupt(value)
intr.resumable      # bool: True if graph can be resumed
intr.ns             # namespace tuple (for subgraph interrupts)
intr.when           # "during" (standard) or "after" (static after-interrupt)
```

---

## Gotchas

1. **Interrupts require a checkpointer.** Without one, `interrupt()` raises immediately. Use `InMemorySaver()` for dev.

2. **Don't pass `Command(update=...)` to resume a regular conversation turn.** `Command(resume=...)` is for resuming interrupted nodes. For a new turn, just pass the new input dict.

3. **The node re-executes from the top on resume.** This is by design. Side effects (DB writes, emails sent) will fire again. Either move them after `interrupt()` or make them idempotent (check-then-write).

4. **Non-deterministic `interrupt()` ordering breaks multi-interrupt nodes.** If `interrupt()` calls are inside `if/else` branches that could execute in different orders across resume attempts, the index-based matching breaks. Always keep interrupt calls at the top level of the function in a fixed order.

5. **`Interrupt` (capital I) is a type; `interrupt()` (lowercase) is the function.** Don't confuse them.

---

# 15 — LangGraph Checkpointers (Persistence)

Checkpointers save a snapshot of the full graph state after every super-step. They enable:
- **Conversation memory** — state survives across invocations on the same `thread_id`
- **Human-in-the-loop** — pause, wait, resume
- **Time travel** — inspect and replay past states
- **Durable execution** — resume after crashes

---

## Checkpointer interface (`BaseCheckpointSaver`)

All checkpointers implement:
```python
from langgraph.checkpoint.base import BaseCheckpointSaver

# Core sync methods:
ckpt.put(config, checkpoint, metadata, new_versions)   # save checkpoint
ckpt.put_writes(config, writes, task_id)               # save pending writes
ckpt.get_tuple(config) -> CheckpointTuple | None       # get specific checkpoint
ckpt.list(config, *, filter=None, before=None, limit=None)  # list checkpoints

# Core async methods (same names with 'a' prefix):
await ckpt.aput(...)
await ckpt.aput_writes(...)
await ckpt.aget_tuple(...)
async for cp in ckpt.alist(...): ...

# Delete a thread:
ckpt.delete_thread(thread_id)          # removes all checkpoints for a thread
```

---

## `InMemorySaver` (dev only)

```python
from langgraph.checkpoint.memory import InMemorySaver

ckpt = InMemorySaver()
graph = builder.compile(checkpointer=ckpt)
```

Lost on process restart. OK for single-process dev/testing.

---

## `SqliteSaver` (local persistence)

```python
from langgraph.checkpoint.sqlite import SqliteSaver

# Context manager (recommended — handles connection lifecycle):
with SqliteSaver.from_conn_string("checkpoints.sqlite") as ckpt:
    graph = builder.compile(checkpointer=ckpt)
    result = graph.invoke(initial, {"configurable": {"thread_id": "t1"}})
    # checkpoints persisted to SQLite file

# Async version (for async code):
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
async with AsyncSqliteSaver.from_conn_string("checkpoints.sqlite") as ckpt:
    graph = builder.compile(checkpointer=ckpt)
    result = await graph.ainvoke(initial, config)
```

SQLite limitation: single-writer. Not for multi-process/multi-instance deployments.

---

## `PostgresSaver` (production)

```python
from langgraph.checkpoint.postgres import PostgresSaver

# Context manager:
conn_string = "postgresql://user:password@host:5432/dbname"
with PostgresSaver.from_conn_string(conn_string) as ckpt:
    ckpt.setup()    # run migrations (ONCE — creates checkpoint tables)
    graph = builder.compile(checkpointer=ckpt)
    result = graph.invoke(initial, {"configurable": {"thread_id": "t1"}})

# Async (recommended for production async workloads):
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
async with AsyncPostgresSaver.from_conn_string(conn_string) as ckpt:
    await ckpt.asetup()
    graph = builder.compile(checkpointer=ckpt)
    result = await graph.ainvoke(initial, config)

# From existing psycopg connection:
import psycopg
with psycopg.connect(conn_string) as conn:
    ckpt = PostgresSaver(conn)
    ckpt.setup()
    ...
```

`ckpt.setup()` / `await ckpt.asetup()` creates the required tables. Call it once at startup. It's idempotent (safe to call multiple times).

---

## `thread_id` and checkpoint addressing

Every invocation that involves a checkpointer MUST include a `thread_id`:

```python
config = {
    "configurable": {
        "thread_id": "user-42-session-abc",    # required
        "checkpoint_id": None,                  # optional: target a specific checkpoint
        "checkpoint_ns": "",                    # optional: namespace for subgraph isolation
    }
}
```

`thread_id` semantics:
- A thread is a sequence of related invocations (a "conversation").
- Same `thread_id` → same thread → history is preserved.
- Different `thread_id` → completely separate history.
- Recommended naming: `"{user_id}:{session_id}"` or similar.

`checkpoint_id`: if provided, starts execution from that specific checkpoint (for time travel / replay).

`checkpoint_ns`: used internally to namespace subgraph checkpoints. Leave blank unless you know what you're doing.

---

## Checkpoint encryption

```python
from langgraph.checkpoint.serde.encrypted import EncryptedSerializer

# AES-256 encryption for checkpoint data at rest:
from Crypto.Cipher import AES
key = b"exactly-32-bytes-secret-key-here"

serde = EncryptedSerializer.from_pycryptodome_aes(key)
with PostgresSaver.from_conn_string(conn_string, serde=serde) as ckpt:
    ckpt.setup()
    graph = builder.compile(checkpointer=ckpt)
```

Default serde is `JsonPlusSerializer` (ormsgpack + JSON fallback, handles all LangChain types, datetimes, enums).

---

## Thread management

```python
# Delete all checkpoints for a thread:
ckpt.delete_thread("user-42-session-abc")

# List all checkpoints for a thread (for debugging):
for cp in ckpt.list({"configurable": {"thread_id": "t1"}}):
    print(cp.checkpoint["id"], cp.metadata["step"], cp.metadata["created_at"])
```

---

# 16 — LangGraph Time Travel

With a checkpointer attached, every invocation creates a checkpoint. You can inspect history, replay from a past state, or fork into a new branch.

---

## `get_state` — current state snapshot

```python
cfg = {"configurable": {"thread_id": "t1"}}

snapshot = graph.get_state(cfg)
# snapshot is a StateSnapshot:
# .values       — the state dict at this checkpoint
# .next         — tuple of node names that will run next (empty if finished)
# .config       — the config dict that uniquely identifies this checkpoint
# .metadata     — {"step": N, "source": "loop", "writes": {...}, "parents": {...}}
# .created_at   — ISO timestamp
# .parent_config — config of the previous checkpoint

print(snapshot.values["messages"][-1].content)
print(snapshot.next)        # e.g., ("tools",) if paused before tools
```

---

## `get_state_history` — full checkpoint history

Returns checkpoints in reverse-chronological order:

```python
for snap in graph.get_state_history(cfg):
    print(snap.metadata["step"], snap.next, snap.created_at)
    # Use snap.config as a time-travel config to replay from this point
```

Get a specific past snapshot:
```python
history = list(graph.get_state_history(cfg))
third_step = history[-3]   # third from the beginning (end of list = oldest)
```

---

## Replaying from a past checkpoint

Use a historical snapshot's `config` as the invocation config:

```python
# Get a past snapshot:
history = list(graph.get_state_history(cfg))
past_snap = history[2]   # some past point

# Replay: re-execute from this checkpoint forward (same inputs, new execution):
result = graph.invoke(None, past_snap.config)
```

Replaying is idempotent — it re-runs the graph from that checkpoint using the same state, but produces a NEW branch (new checkpoints in the same thread).

---

## `update_state` — fork / edit past state

Edit a checkpoint to create a new branch or inject state:

```python
# Fork: change a value in a past state:
fork_config = graph.update_state(
    past_snap.config,
    values={"topic": "machine learning"},    # partial state update
    as_node="topic_selector",               # attribute update to this node
)
# fork_config points to the new forked checkpoint

# Resume execution from the fork:
result = graph.invoke(None, fork_config)
```

`as_node` parameter:
- Tells LangGraph to treat the update as if `topic_selector` node produced it.
- Determines the "next" node: whatever would follow `topic_selector` in the graph.
- If omitted, the current checkpoint's "next" node is used.

**Common use cases:**
1. **Testing:** inject specific state to test a particular code path.
2. **Correction:** user spots an error; edit the bad message and replay.
3. **HITL loop:** pause, get feedback, edit state with the feedback, resume.
4. **Debugging:** fork at a suspicious step, try alternative inputs.

```python
# HITL edit-and-resume pattern:
# 1. Pause at interrupt or interrupt_before:
out = graph.invoke(input, cfg)

# 2. User reviews and edits the last AI message:
state = graph.get_state(cfg)
last_msg = state.values["messages"][-1]
corrected_msg = AIMessage(content="Corrected answer.", id=last_msg.id)  # same ID = replace

graph.update_state(cfg, {"messages": [corrected_msg]})

# 3. Resume:
result = graph.invoke(None, cfg)
```

---

## Async versions

```python
snapshot = await graph.aget_state(cfg)
async for snap in graph.aget_state_history(cfg): ...
fork_config = await graph.aupdate_state(cfg, values={"key": "val"}, as_node="node")
```

---

## Limitations

1. **`update_state` cannot disambiguate between simultaneously-updated parallel branches** (e.g., two Send workers that both update the same key). Raises `InvalidUpdateError`. Only single-branch updates are supported.
2. **`as_node` must be a real node in the graph.** Typos silently create invalid forks.
3. **History is stored per-thread.** Switching `thread_id` gives you a different history.
4. **Replaying re-runs ALL side effects** (API calls, DB writes, etc.) from the checkpoint forward. Design side effects to be idempotent.
