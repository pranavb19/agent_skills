# 10 — LLM Caching

LangChain supports process-level LLM response caching. When the same `(prompt, llm_string)` pair is seen again, the cached response is returned without making an API call.

## Enabling cache globally

```python
from langchain.globals import set_llm_cache, get_llm_cache

# In-memory (lost on process restart, good for dev):
from langchain_community.cache import InMemoryCache
set_llm_cache(InMemoryCache())

# SQLite (persists to disk, good for dev/local testing):
from langchain_community.cache import SQLiteCache
set_llm_cache(SQLiteCache(database_path=".langchain_cache.db"))

# Check current cache:
current = get_llm_cache()

# Disable caching:
set_llm_cache(None)
```

## Cache backends

### `InMemoryCache`
```python
from langchain_community.cache import InMemoryCache
set_llm_cache(InMemoryCache())
# Exact match on (prompt_string, llm_string). Cleared on process restart.
```

### `SQLiteCache`
```python
from langchain_community.cache import SQLiteCache
set_llm_cache(SQLiteCache(database_path=".langchain.db"))
# Persists across restarts. Good for test suites and dev.
```

### `RedisCache` (production exact match)
```python
from langchain_community.cache import RedisCache
import redis
redis_client = redis.Redis(host="localhost", port=6379, db=0)
set_llm_cache(RedisCache(redis_=redis_client, ttl=3600))  # ttl in seconds
```

### `RedisSemanticCache` (production semantic match)
Returns cached responses for semantically similar prompts, not just exact matches:
```python
from langchain_community.cache import RedisSemanticCache
from langchain_google_genai import GoogleGenerativeAIEmbeddings

set_llm_cache(RedisSemanticCache(
    redis_url="redis://localhost:6379",
    embedding=GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001", task_type="SEMANTIC_SIMILARITY"
    ),
    score_threshold=0.2,    # lower = stricter match
))
```

### `GPTCache` (advanced, multi-backend)
```python
from langchain_community.cache import GPTCache
import gptcache
set_llm_cache(GPTCache())   # supports local/cloud backends, semantic + exact
```

## Limitations and gotchas

1. **Caching does NOT apply to streaming.** `.stream()` / `.astream()` calls bypass the cache even if a cache is set. Only `.invoke()` / `.ainvoke()` / `.batch()` use the cache.

2. **Cache key includes `llm_string`.** Changing model parameters (temperature, max_tokens) creates a different `llm_string` and misses the cache. For evaluation/testing where you want cache hits, keep LLM parameters fixed.

3. **Semantic cache false positives.** `RedisSemanticCache` may return cached responses for prompts that are semantically similar but require different answers. Tune `score_threshold` carefully and monitor for quality regressions.

4. **SQLiteCache is single-writer.** In multi-process deployments (e.g., gunicorn with multiple workers), concurrent writes can cause database locking issues. Use `RedisCache` for multi-process scenarios.

5. **Cache hits bypass LangSmith logging.** Cached responses don't generate new LangSmith traces (no API call was made). If you need to audit every invocation, set `callbacks=[LangSmithCallbackHandler()]` explicitly to force trace logging regardless.

## Per-chain cache disabling

```python
# Disable caching for a specific chain, even if global cache is set:
chain = prompt | llm.bind(cache=False) | parser
```

---

# 11 — Memory (`RunnableWithMessageHistory`)

> **Important:** For production-grade conversation memory in agents, use LangGraph's checkpointer and `MessagesState` (see `12-langgraph-stategraph.md` and `15-langgraph-checkpointers.md`). `RunnableWithMessageHistory` is appropriate for simpler LCEL chains where you don't need LangGraph's full graph runtime.

## `RunnableWithMessageHistory`

Wraps an LCEL chain so that, given a `session_id`, it automatically:
1. Loads existing messages from a `BaseChatMessageHistory` backend.
2. Appends them to the prompt's `MessagesPlaceholder`.
3. After the chain runs, saves the new human + AI messages back to the backend.

```python
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser

# ── 1. History backend (in-memory for dev) ────────────────────────────
store: dict[str, InMemoryChatMessageHistory] = {}

def get_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# ── 2. Chain ───────────────────────────────────────────────────────────
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Be concise."),
    MessagesPlaceholder("history"),    # MUST match history_messages_key below
    ("human", "{question}"),           # MUST match input_messages_key below
])
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
chain = prompt | llm | StrOutputParser()

# ── 3. Wrap with history ───────────────────────────────────────────────
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_history,
    input_messages_key="question",       # key in the input dict that is the new human message
    history_messages_key="history",      # key in the prompt for injecting history
    # output_messages_key="answer",      # if chain returns dict, key of the AI output
)

# ── 4. Invoke ──────────────────────────────────────────────────────────
config = {"configurable": {"session_id": "user-1"}}

r1 = chain_with_history.invoke({"question": "My name is Alice."}, config=config)
r2 = chain_with_history.invoke({"question": "What's my name?"}, config=config)
print(r2)  # "Your name is Alice."
```

### Multiple config keys

If you need more than `session_id` to look up history (e.g., `user_id` + `conversation_id`):

```python
chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda user_id, conv_id: get_history_by_user(user_id, conv_id),
    input_messages_key="question",
    history_messages_key="history",
    history_factory_config=[
        ConfigurableFieldSpec(id="user_id", annotation=str),
        ConfigurableFieldSpec(id="conv_id", annotation=str),
    ],
)

config = {"configurable": {"user_id": "u123", "conv_id": "c456"}}
chain_with_history.invoke({"question": "..."}, config=config)
```

## `BaseChatMessageHistory` backends

```python
from langchain_core.chat_history import InMemoryChatMessageHistory  # dev

# Persistent backends (from langchain_community.chat_message_histories):
from langchain_community.chat_message_histories import (
    FileChatMessageHistory,           # JSON files
    SQLChatMessageHistory,            # any SQLAlchemy DB
    PostgresChatMessageHistory,       # Postgres
    RedisChatMessageHistory,          # Redis
    MongoDBChatMessageHistory,        # MongoDB
    DynamoDBChatMessageHistory,       # AWS DynamoDB
    FirestoreChatMessageHistory,      # GCP Firestore
    ZepChatMessageHistory,            # Zep memory server
)

# File backend:
from langchain_community.chat_message_histories import FileChatMessageHistory
get_history = lambda sid: FileChatMessageHistory(f"history_{sid}.json")

# Redis backend:
from langchain_community.chat_message_histories import RedisChatMessageHistory
def get_redis_history(session_id: str):
    return RedisChatMessageHistory(
        session_id, url="redis://localhost:6379", ttl=3600
    )
```

## When to use `RunnableWithMessageHistory` vs LangGraph

| Scenario | Use |
|---|---|
| Simple LCEL chain, no tool calling | `RunnableWithMessageHistory` |
| Agent with tools | LangGraph checkpointer (via `create_agent` or `StateGraph`) |
| Need time travel / state editing | LangGraph checkpointer |
| Need cross-thread memory (store) | LangGraph `Store` |
| Need human-in-the-loop | LangGraph interrupts |
| Need complex multi-step flow | LangGraph |
| Simple chatbot, minimal dependencies | `RunnableWithMessageHistory` |

## History message trimming

For long conversations, trim to the most recent N messages before injecting into the prompt:

```python
from langchain_core.messages import trim_messages, SystemMessage

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are helpful."),
    MessagesPlaceholder("history"),
    ("human", "{question}"),
])

chain = (
    RunnablePassthrough.assign(
        history=lambda d: trim_messages(
            d["history"],
            max_tokens=4096,
            token_counter=ChatGoogleGenerativeAI(model="gemini-2.5-flash"),
            strategy="last",              # keep the most recent messages
            start_on="human",             # ensure first kept message is from human
            include_system=True,          # never trim the system message
        )
    )
    | prompt | llm | StrOutputParser()
)
```

`trim_messages` parameters: `max_tokens` (or `max_messages`), `strategy` (`"last"` or `"first"`), `token_counter` (model or callable), `start_on` (`"human"` or `"ai"`), `include_system` (bool), `allow_partial` (bool — split messages at token boundary).
