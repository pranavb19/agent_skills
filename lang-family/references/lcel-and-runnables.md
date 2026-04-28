# 04 — LCEL and the Runnable Interface

LangChain Expression Language (LCEL) is the declarative composition layer. Every primitive (model, prompt, parser, retriever, tool, custom function) implements `Runnable`, so they all share the same invocation interface and can be connected with `|`.

---

## The `Runnable` Interface

Every `Runnable` implements these methods (sync + async versions):

```python
# ── Single invocation ───────────────────────────────────────────────
result   = runnable.invoke(input, config=None)
result   = await runnable.ainvoke(input, config=None)

# ── Streaming ───────────────────────────────────────────────────────
for chunk in runnable.stream(input, config=None):         ...
async for chunk in runnable.astream(input, config=None):  ...

# Fine-grained event stream (use this for agent streaming UIs):
async for event in runnable.astream_events(input, version="v2", config=None,
                                           include_types=["chat_model"],
                                           exclude_names=["retriever"]): ...

# ── Batching ────────────────────────────────────────────────────────
results = runnable.batch([input1, input2, ...], config=None)
results = await runnable.abatch([input1, input2, ...], config=None)

# ── Composition helpers ─────────────────────────────────────────────
new_r = runnable.with_retry(stop_after_attempt=3, wait_exponential_jitter=True)
new_r = runnable.with_fallbacks([fallback1, fallback2])
new_r = runnable.with_config(run_name="step1", tags=["prod"], metadata={"v": "1.0"})
new_r = runnable.with_listeners(on_start=..., on_end=..., on_error=...)

# ── Introspection ───────────────────────────────────────────────────
schema = runnable.input_schema     # Pydantic model for expected input
schema = runnable.output_schema    # Pydantic model for output
```

**Input/output types:**
- A chat model accepts `str | list[BaseMessage] | PromptValue` and outputs `AIMessage`.
- A parser accepts `AIMessage | AIMessageChunk | str` and outputs the parsed type.
- A prompt accepts `dict` and outputs `PromptValue`.
- A chain (sequence) accepts whatever the first step accepts and outputs whatever the last step outputs.

---

## The Pipe Operator (`|`) and `RunnableSequence`

`|` builds a `RunnableSequence`: output of step N becomes input of step N+1.

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
prompt = ChatPromptTemplate.from_messages([("human", "Summarize: {text}")])

chain = prompt | llm | StrOutputParser()
result: str = chain.invoke({"text": "Long article..."})
```

Under the hood: `prompt | llm` creates a `RunnableSequence([prompt, llm])`. Adding `| StrOutputParser()` extends the sequence.

**Type coercion in the pipe chain:**
- A plain Python `dict` literal becomes a `RunnableParallel` when piped (see below).
- A plain Python function is NOT auto-coerced — wrap it in `RunnableLambda`.

---

## `RunnableParallel` — run branches in parallel

Runs multiple runnables with the same input and combines outputs into a dict.

```python
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

# Dict literal inside a chain is auto-coerced to RunnableParallel:
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Explicit construction (equivalent):
parallel = RunnableParallel(
    context=retriever,
    question=RunnablePassthrough(),
)

# Explicit class:
from langchain_core.runnables import RunnableParallel
parallel = RunnableParallel({
    "summary": summarize_chain,
    "keywords": keyword_chain,
    "sentiment": sentiment_chain,
})
result = parallel.invoke({"text": "Some document..."})
# result = {"summary": "...", "keywords": [...], "sentiment": "positive"}
```

Parallel branches run concurrently (async) when using `ainvoke` / `astream`. For `invoke` (sync), they run sequentially unless the runnable itself uses threads internally.

---

## `RunnablePassthrough` — identity / augment

Passes input through unchanged. The `.assign()` method adds computed fields while preserving existing ones.

```python
from langchain_core.runnables import RunnablePassthrough

# Pure passthrough — output = input
passthrough = RunnablePassthrough()
passthrough.invoke({"x": 1})   # → {"x": 1}

# Assign: add new keys computed from the input while keeping original keys
augmented = RunnablePassthrough.assign(
    word_count=lambda d: len(d["text"].split()),
    upper=lambda d: d["text"].upper(),
)
augmented.invoke({"text": "hello world"})
# → {"text": "hello world", "word_count": 2, "upper": "HELLO WORLD"}

# .assign() on a chain adds fields mid-chain:
chain = (
    {"raw_doc": RunnablePassthrough(), "question": RunnablePassthrough()}
    | RunnablePassthrough.assign(context=lambda d: format_docs(d["raw_doc"]))
    | prompt
    | llm
)
```

---

## `RunnableLambda` — wrap a Python function

Wraps a **single-argument** Python function (sync or async). The one argument can be a dict — unpack inside.

```python
from langchain_core.runnables import RunnableLambda

# Sync
def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

formatter = RunnableLambda(format_docs)

# Async
async def async_lookup(query: str) -> str:
    return await db.get(query)

lookup = RunnableLambda(async_lookup)   # works in astream/ainvoke contexts

# With config (access RunnableConfig inside the function):
def my_step(input, config):     # second arg MUST be named "config"
    run_name = config.get("run_name", "unknown")
    return {"result": run_name}

step = RunnableLambda(my_step)

# Inline with | (using the decorator approach):
chain = retriever | RunnableLambda(format_docs) | prompt | llm
```

**Gotcha:** `RunnableLambda` accepts exactly one positional argument. For multi-arg logic, pack state into a dict and unpack inside.

---

## `RunnableBranch` — conditional routing

Routes to different chains based on a condition. Evaluated in order; first matching condition wins. A default (no condition) must be last.

```python
from langchain_core.runnables import RunnableBranch

branch = RunnableBranch(
    (lambda d: d["topic"] == "math",    math_chain),
    (lambda d: d["topic"] == "science", science_chain),
    general_chain,   # default — no condition tuple, just the runnable
)

result = branch.invoke({"topic": "math", "question": "What is calculus?"})
```

**Alternative (functional) routing with `RunnableLambda`:**
```python
def router(state: dict):
    topic = state["topic"]
    if topic == "math":
        return math_chain.invoke(state)
    elif topic == "science":
        return science_chain.invoke(state)
    return general_chain.invoke(state)

branch = RunnableLambda(router)
```

The `RunnableLambda` approach is more flexible (arbitrary Python logic) but loses the declarative graph structure. Use `RunnableBranch` when you want introspection/serialization; use `RunnableLambda` for complex routing logic.

---

## `RunnableConfig` — threading config through chains

A dict (or typed dict) that flows through every `invoke` / `stream` call:

```python
from langchain_core.runnables import RunnableConfig

config: RunnableConfig = {
    "run_name": "my-pipeline",       # Appears in LangSmith traces
    "tags": ["prod", "v1.2"],        # Appear in LangSmith traces
    "metadata": {"user_id": "u123"}, # Appear in LangSmith traces
    "callbacks": [...],              # Callback handlers (e.g., streaming)
    "max_concurrency": 4,            # For .batch() parallelism
    "recursion_limit": 25,           # For LangGraph (default 25)
    "configurable": {                # Arbitrary runtime config your code reads
        "thread_id": "session-abc",  # Required when using LangGraph checkpointers
        "temperature": 0.5,          # Can be used by model.configurable_fields()
    },
}

chain.invoke({"question": "..."}, config=config)
```

### `configurable_fields` — user-configurable model parameters

```python
from langchain_core.runnables import ConfigurableField

llm_configurable = llm.configurable_fields(
    temperature=ConfigurableField(
        id="temperature",
        name="LLM Temperature",
        description="The sampling temperature",
    )
)

# Now override temperature at invocation time:
chain = prompt | llm_configurable | StrOutputParser()
result = chain.invoke(
    {"question": "Be creative!"},
    config={"configurable": {"temperature": 1.5}},
)
```

### `configurable_alternatives` — swap entire components

```python
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash").configurable_alternatives(
    ConfigurableField(id="llm"),
    default_key="gemini",
    anthropic=ChatAnthropic(model="claude-sonnet-4-5"),
)

chain = prompt | llm | StrOutputParser()
# Use default (Gemini):
chain.invoke({"q": "..."})
# Swap to Anthropic at runtime:
chain.invoke({"q": "..."}, config={"configurable": {"llm": "anthropic"}})
```

---

## `.with_retry()` and `.with_fallbacks()`

```python
# Retry on transient errors (rate limits, timeouts):
resilient = llm.with_retry(
    stop_after_attempt=3,
    wait_exponential_jitter=True,    # adds jitter to exponential backoff
    retry_if_exception_type=(TimeoutError, ConnectionError),
)

# Fallback to a different model on failure:
primary = ChatGoogleGenerativeAI(model="gemini-2.5-pro")
cheap   = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
with_fallback = primary.with_fallbacks([cheap])

chain = prompt | with_fallback | StrOutputParser()
```

---

## `.bind()` — pre-bind kwargs

Useful for pre-setting model parameters like `stop`, `tool_choice`, or tool bindings:

```python
# Pre-bind stop sequences:
llm_stopped = llm.bind(stop=["###"])

# Pre-bind tools (same as bind_tools):
llm_with_tools = llm.bind(tools=[tool_schema_dict])

# In a chain:
chain = prompt | llm.bind(stop=["END"]) | parser
```

---

## `.assign()` — add fields mid-chain

```python
chain = (
    RunnablePassthrough.assign(
        docs=lambda d: retriever.invoke(d["question"]),
    )
    | RunnablePassthrough.assign(
        context=lambda d: "\n\n".join(doc.page_content for doc in d["docs"]),
    )
    | prompt
    | llm
    | StrOutputParser()
)
chain.invoke({"question": "What is RAG?"})
```

---

## Streaming patterns

```python
# ── Basic streaming (receives chunks of the final output type) ──────
for chunk in chain.stream({"question": "..."}):
    print(chunk, end="", flush=True)

# ── astream_events (fine-grained, recommended for UIs) ──────────────
async for event in chain.astream_events({"question": "..."}, version="v2"):
    # event keys: "event", "name", "run_id", "parent_ids", "tags",
    #             "metadata", "data"
    if event["event"] == "on_chat_model_stream":
        chunk = event["data"]["chunk"]
        if chunk.content:
            print(chunk.content, end="", flush=True)
    elif event["event"] == "on_retriever_end":
        docs = event["data"]["output"]
        print(f"Retrieved {len(docs)} docs")
```

Event types for `astream_events`:
- `on_chain_start/end/stream/error` — for LCEL chains, `RunnableSequence`, etc.
- `on_chat_model_start/end/stream/error` — for chat model calls.
- `on_llm_start/end/stream/error` — for text LLM calls (legacy).
- `on_tool_start/end/error` — for tool calls.
- `on_retriever_start/end/error` — for retriever calls.
- `on_custom_event` — for `adispatch_custom_event`.

Filtering:
```python
async for event in chain.astream_events(
    input,
    version="v2",
    include_names=["ChatGoogleGenerativeAI"],   # only events from this runnable name
    include_types=["chat_model"],               # or by type
    include_tags=["llm-call"],                  # or by tags
    exclude_names=[...],                        # or exclude
):
    ...
```

---

## Batch execution

```python
# Parallel batch (uses max_concurrency from config):
results = chain.batch(
    [{"question": "Q1"}, {"question": "Q2"}, {"question": "Q3"}],
    config={"max_concurrency": 5},  # default: unlimited
    return_exceptions=True,         # return exceptions instead of raising
)

# Async batch:
results = await chain.abatch([...], config={"max_concurrency": 5})
```

Use batch over a Python loop whenever you have many independent inputs — it's typically 3–10x faster due to parallelism.

---

## Disabling streaming on a chain

```python
# If a downstream component doesn't support streaming, disable it:
chain = prompt | llm.bind(disable_streaming=True) | parser
```

---

## Complete LCEL patterns reference

### Pattern 1: Basic LLM chain

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

chain = (
    ChatPromptTemplate.from_messages([("human", "{question}")])
    | ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    | StrOutputParser()
)
```

### Pattern 2: RAG chain

```python
from langchain_core.runnables import RunnablePassthrough

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```

### Pattern 3: Parallel analysis

```python
from langchain_core.runnables import RunnableParallel

analysis = RunnableParallel(
    summary=summary_chain,
    sentiment=sentiment_chain,
    topics=topics_chain,
)
```

### Pattern 4: Conditional routing

```python
from langchain_core.runnables import RunnableBranch

route = RunnableBranch(
    (lambda x: x["intent"] == "question", qa_chain),
    (lambda x: x["intent"] == "task",     task_chain),
    fallback_chain,
)
```

### Pattern 5: Dynamic augmentation

```python
chain = (
    RunnablePassthrough.assign(retrieved=lambda d: retriever.invoke(d["query"]))
    | RunnablePassthrough.assign(context=lambda d: format(d["retrieved"]))
    | prompt | llm | StrOutputParser()
)
```

---

## Common gotchas

1. **Plain functions need `RunnableLambda`** — `chain = prompt | llm | my_function` will raise a `TypeError`. Write `| RunnableLambda(my_function)`.

2. **Dict literal is a `RunnableParallel`** — `{"a": runnable, "b": passthrough}` in a pipe chain is coerced to `RunnableParallel`. A plain dict with static values is NOT coerced — use `RunnableLambda(lambda _: {"a": 1})`.

3. **`RunnableLambda` accepts exactly one argument** — if you need multiple values, receive `input: dict` and unpack inside.

4. **`.assign()` returns a new runnable** — it doesn't mutate. Chain its result: `chain = RunnablePassthrough.assign(...) | next_step`.

5. **`with_fallbacks` swallows all exceptions from the primary** — make sure the exception types you want to handle are actually raised (not caught inside the model class) or use `exceptions_to_handle=` parameter.

6. **`configurable_fields` requires a fresh invoke config** — the config must be passed to every `.invoke()` call; it doesn't persist across calls.

7. **For `astream_events`, always pass `version="v2"`** — v1 is deprecated and will be removed.
