# 09 — Streaming and Async

Every `Runnable` in LangChain exposes a consistent set of streaming and async methods. This reference covers the APIs, event types, and patterns for building responsive real-time UIs and high-throughput batch pipelines.

---

## Methods overview

| Method | Description | Returns |
|---|---|---|
| `.invoke(input, config)` | Sync, single call | full output |
| `.ainvoke(input, config)` | Async, single call | full output |
| `.stream(input, config)` | Sync, stream output chunks | iterator of output chunks |
| `.astream(input, config)` | Async, stream output chunks | async iterator of output chunks |
| `.astream_events(input, version, config, ...)` | Async, fine-grained events for every step | async iterator of event dicts |
| `.batch(inputs, config)` | Sync, multiple inputs in parallel | list of outputs |
| `.abatch(inputs, config)` | Async, multiple inputs in parallel | list of outputs |

---

## `.stream` and `.astream`

Emit chunks of the **final output type**. For a chain ending with `StrOutputParser`, emits string fragments. For a chain ending with a chat model, emits `AIMessageChunk` objects.

```python
# Sync stream:
for chunk in chain.stream({"question": "Explain quantum entanglement."}):
    print(chunk, end="", flush=True)

# Async stream:
async def stream_response():
    async for chunk in chain.astream({"question": "Explain quantum entanglement."}):
        print(chunk, end="", flush=True)

# Only the last step streams — earlier steps complete before streaming starts.
# Exception: if all steps support streaming, intermediate steps also stream chunks.
```

**Streaming from a chat model (no parser):**
```python
async for chunk in llm.astream("Tell me a story."):
    print(chunk.content, end="", flush=True)
    # chunk is AIMessageChunk with .content (str), .tool_call_chunks, etc.
```

**Disable streaming (for components that don't support it):**
```python
chain = prompt | llm.bind(disable_streaming=True) | parser
```

---

## `.astream_events` — fine-grained event stream (recommended for UIs)

The most powerful streaming API. Emits an event for every step's start, stream, and end. Essential for building streaming UIs that show progress, tool calls, and token output simultaneously.

Always use `version="v2"` (v1 is deprecated):

```python
async def stream_with_events(input_data: dict):
    async for event in chain.astream_events(input_data, version="v2"):
        event_type = event["event"]
        name      = event["name"]          # runnable name
        run_id    = event["run_id"]        # UUID of this run
        parent_ids= event["parent_ids"]    # parent run IDs
        tags      = event["tags"]          # from RunnableConfig
        metadata  = event["metadata"]      # from RunnableConfig
        data      = event["data"]          # event-specific payload

        if event_type == "on_chat_model_stream":
            chunk = data["chunk"]          # AIMessageChunk
            if chunk.content:
                print(chunk.content, end="", flush=True)

        elif event_type == "on_tool_start":
            print(f"\n[Tool] {name}({data['input']})")

        elif event_type == "on_tool_end":
            print(f"\n[Tool result] {data['output']}")

        elif event_type == "on_retriever_end":
            print(f"\n[Retrieved {len(data['output'])} docs]")
```

### All event types

| Event | When fired | `data` contents |
|---|---|---|
| `on_chain_start` | Chain/runnable begins | `{"input": ...}` |
| `on_chain_stream` | Intermediate chunk from chain | `{"chunk": ...}` |
| `on_chain_end` | Chain/runnable completes | `{"input": ..., "output": ...}` |
| `on_chain_error` | Chain/runnable errors | `{"input": ..., "error": ...}` |
| `on_chat_model_start` | LLM begins | `{"input": {"messages": [...]}}` |
| `on_chat_model_stream` | Token streamed | `{"chunk": AIMessageChunk}` |
| `on_chat_model_end` | LLM completes | `{"input": ..., "output": AIMessage}` |
| `on_llm_start/stream/end/error` | Legacy text LLM events | similar |
| `on_tool_start` | Tool begins | `{"input": tool_args_dict}` |
| `on_tool_end` | Tool completes | `{"output": tool_result}` |
| `on_tool_error` | Tool errors | `{"error": exception}` |
| `on_retriever_start` | Retriever begins | `{"input": {"query": "..."}}` |
| `on_retriever_end` | Retriever completes | `{"output": list[Document]}` |
| `on_retriever_error` | Retriever errors | `{"error": exception}` |
| `on_custom_event` | Custom event via `adispatch_custom_event` | user-defined |
| `on_prompt_start/end` | Prompt template | `{"input": ..., "output": PromptValue}` |
| `on_parser_start/end` | Output parser | `{"input": ..., "output": ...}` |

### Filtering events

```python
async for event in chain.astream_events(
    input_data, version="v2",
    # Include only matching events (AND logic within a filter param):
    include_names=["ChatGoogleGenerativeAI"],    # by runnable .name
    include_types=["chat_model", "tool"],         # by type string
    include_tags=["my-tag"],                      # by tag (from RunnableConfig)
    # Exclude:
    exclude_names=["intermediate_parser"],
    exclude_types=["chain"],
    exclude_tags=["internal"],
):
    ...
```

### Custom events (emit from inside any function)

```python
from langchain_core.callbacks.manager import adispatch_custom_event, dispatch_custom_event

# From an async function:
async def processing_step(data: str, config) -> str:
    await adispatch_custom_event("progress", {"pct": 0, "msg": "Starting"}, config=config)
    result = await do_work(data)
    await adispatch_custom_event("progress", {"pct": 100, "msg": "Done"}, config=config)
    return result

# From a sync function:
def sync_step(data: str, config) -> str:
    dispatch_custom_event("sync_progress", {"step": "begin"}, config=config)
    return process(data)

# Consumer:
async for event in chain.astream_events(input, version="v2"):
    if event["event"] == "on_custom_event" and event["name"] == "progress":
        print(f"Progress: {event['data']['pct']}% — {event['data']['msg']}")
```

**Gotcha:** `adispatch_custom_event` requires the `config` to be passed through from the `Runnable`'s invocation. If the function is inside `RunnableLambda`, access config as the second argument:

```python
from langchain_core.runnables import RunnableLambda

async def my_fn(input, config):  # second arg MUST be named 'config'
    await adispatch_custom_event("my-event", {"data": input}, config=config)
    return input

step = RunnableLambda(my_fn)
```

---

## Streaming from LangGraph

LangGraph streaming is covered in depth in `20-langgraph-streaming.md`. In summary:

```python
# stream_mode="messages" — token-level streaming from any LLM node
for chunk in graph.stream(input, stream_mode="messages"):
    msg, metadata = chunk
    if hasattr(msg, "content") and msg.content:
        print(msg.content, end="")

# stream_mode="updates" — node-level state deltas
for chunk in graph.stream(input, stream_mode="updates"):
    for node_name, delta in chunk.items():
        print(f"{node_name}: {list(delta.keys())}")

# Unified v2 format (recommended):
for chunk in graph.stream(input, stream_mode=["messages", "updates"], version="v2"):
    if chunk["type"] == "messages":
        msg, _ = chunk["data"]
        if msg.content: print(msg.content, end="")
    elif chunk["type"] == "updates":
        ...
```

---

## Batch execution

`.batch` / `.abatch` run multiple inputs concurrently. Prefer over Python loops for throughput.

```python
inputs = [
    {"question": "Q1"},
    {"question": "Q2"},
    {"question": "Q3"},
]

# Sync batch:
results = chain.batch(inputs, config={"max_concurrency": 5},
                      return_exceptions=True)

# Async batch:
results = await chain.abatch(inputs, config={"max_concurrency": 5},
                             return_exceptions=True)

# With per-input config (pass a list of configs):
configs = [
    {"metadata": {"input_id": i}} for i in range(len(inputs))
]
results = chain.batch(inputs, config=configs)
```

`return_exceptions=True` means failed inputs return `Exception` objects instead of raising; the other results are still returned. Useful for bulk processing where some inputs may fail.

`max_concurrency` in `RunnableConfig` controls how many inputs run simultaneously. Default is unlimited (all run in parallel). Tune based on API rate limits.

---

## Async patterns

### Async agent loop (FastAPI / async frameworks)

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()

@app.post("/chat")
async def chat(request: dict):
    async def generate():
        async for event in agent.astream_events(
            {"messages": [{"role": "user", "content": request["message"]}]},
            version="v2",
            include_types=["chat_model"],
        ):
            if event["event"] == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                if chunk.content:
                    yield f"data: {chunk.content}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
```

### Async context manager for checkpointers

```python
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

async def main():
    async with AsyncPostgresSaver.from_conn_string("postgresql://...") as ckpt:
        agent = create_agent(model=llm, tools=tools, checkpointer=ckpt)
        async for chunk in agent.astream(input, stream_mode="messages"):
            msg, _ = chunk
            if msg.content: print(msg.content, end="")
```

---

## Performance tips

1. **Use `abatch` for bulk embedding:** when indexing thousands of documents, `await embeddings.aembed_documents(texts)` plus `await vector_store.aadd_documents(chunks)` runs concurrently across the batch. Pair with `max_concurrency` to stay within rate limits.

2. **Don't use `stream` when you only need the final output.** `stream` requires iterating through all chunks. If you don't need streaming in the UI, `invoke` is simpler and has less overhead.

3. **For LangGraph token streaming**, always use `stream_mode="messages"` (not `"values"`) — it emits token-by-token as the LLM generates, rather than waiting for the node to complete.

4. **`astream_events` has overhead.** It fires events for every step including intermediate chains. For production high-throughput, use `astream(stream_mode="messages")` instead if you only need tokens.

5. **Thread safety for `.stream`:** the sync `.stream` iterator is NOT thread-safe. In multi-threaded applications, create separate chain instances or use the async API.
