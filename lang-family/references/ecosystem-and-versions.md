# 00 — Ecosystem, Versions, and Package Layout

This file is the source of truth for **which package something lives in**, **what's still supported**, and **what was deprecated in the v1.0 transition**. Read this first whenever you're unsure where to import from.

## Stable versions targeted (April 2026)

| Package | Version | Notes |
|---|---|---|
| `langchain` | 1.3.x | The orchestration / agents package. Contains `langchain.agents.create_agent`, middleware, `init_chat_model`. |
| `langchain-core` | 1.x | Foundational abstractions (Runnables, messages, prompts, parsers, tools, documents). Pinned by all other packages. |
| `langchain-community` | latest | Document loaders, legacy retrievers, chat-message-history backends. Optional but often needed. |
| `langgraph` | 1.1.x | Graph runtime, `StateGraph`, prebuilt agents. Has its own minor-version cadence. |
| `langgraph-prebuilt` | latest | Ships with langgraph; provides `create_react_agent`, `ToolNode`, `tools_condition`. |
| `langgraph-checkpoint` | latest | Base checkpointer interface. |
| `langgraph-checkpoint-sqlite` | latest | SQLite checkpointer (dev). |
| `langgraph-checkpoint-postgres` | latest | Postgres checkpointer (prod). |
| `langsmith` | 0.7.x | Tracing client, `@traceable`, `evaluate`, `Client`. |
| `langchain-google-genai` | 4.x | Gemini chat models + embeddings. Uses the consolidated `google-genai` SDK underneath. |
| `langchain-text-splitters` | latest | All text splitters. |
| `langchain-classic` | latest | **Compatibility package only.** Holds deprecated APIs from before v1.0. Don't import from here in new code. |

Python ≥ **3.10** required.

## Recommended `pip install` for a fresh agent project

```bash
pip install -U langchain langchain-core langgraph langsmith
pip install -U langchain-google-genai
pip install -U langchain-text-splitters

# Persistence
pip install -U langgraph-checkpoint-sqlite        # local
pip install -U langgraph-checkpoint-postgres      # production

# RAG
pip install -U langchain-community                # for loaders
pip install -U langchain-chroma                   # local vector store
pip install -U faiss-cpu                          # alternative local vector store

# Multi-agent helpers
pip install -U langgraph-supervisor langgraph-swarm

# Observability (OTel export)
pip install -U "langsmith[otel]"

# Optional: pytest plugin for LLM evals
pip install -U "langsmith[pytest]"
```

## Import map — where to find each abstraction

This is the canonical map. If something doesn't import from where this table says it does, you have an outdated tutorial.

### `langchain-core` (`langchain_core.*`)

| Import | What it is |
|---|---|
| `langchain_core.messages.{HumanMessage, AIMessage, SystemMessage, ToolMessage, AnyMessage}` | Message types. |
| `langchain_core.messages.AIMessageChunk` | Streamed AI message chunk (has `.tool_call_chunks`). |
| `langchain_core.prompts.{PromptTemplate, ChatPromptTemplate, MessagesPlaceholder, FewShotPromptTemplate, FewShotChatMessagePromptTemplate}` | Prompt templates. |
| `langchain_core.runnables.{RunnableSequence, RunnableParallel, RunnableLambda, RunnablePassthrough, RunnableBranch, RunnableConfig, Runnable}` | LCEL primitives. |
| `langchain_core.runnables.history.RunnableWithMessageHistory` | Per-session message-history wrapper. |
| `langchain_core.tools.{tool, BaseTool, StructuredTool}` | Tool abstractions. |
| `langchain_core.output_parsers.{StrOutputParser, JsonOutputParser, PydanticOutputParser, XMLOutputParser}` | Output parsers. |
| `langchain_core.documents.Document` | The unit of retrieval — `(page_content: str, metadata: dict)`. |
| `langchain_core.retrievers.BaseRetriever` | Base class for custom retrievers. |
| `langchain_core.embeddings.Embeddings` | Base class for embedding models. |
| `langchain_core.vectorstores.VectorStore` | Base class for vector stores. |
| `langchain_core.chat_history.{BaseChatMessageHistory, InMemoryChatMessageHistory}` | Chat history backends. |
| `langchain_core.callbacks.manager.{adispatch_custom_event, dispatch_custom_event}` | Emit custom events into the callback/event stream. |

### `langchain` (`langchain.*`) — the v1 orchestration package

| Import | What it is |
|---|---|
| `langchain.agents.create_agent` | **The v1 primary agent factory.** Returns a compiled LangGraph graph. |
| `langchain.agents.middleware.{SummarizationMiddleware, HumanInTheLoopMiddleware, ModelCallLimitMiddleware, ToolCallLimitMiddleware, ModelFallbackMiddleware, PIIMiddleware, TodoListMiddleware, LLMToolSelectorMiddleware, ToolRetryMiddleware, ToolEmulator, ContextEditingMiddleware, ShellToolMiddleware, FilesystemSearchMiddleware, AgentMiddleware}` | Built-in middleware + base class. |
| `langchain.agents.middleware.{before_agent, before_model, wrap_model_call, after_model, wrap_tool_call, after_agent}` | Decorator-style middleware hooks. |
| `langchain.chat_models.init_chat_model` | Provider-neutral model factory: `init_chat_model("google_genai:gemini-2.5-flash")`. |
| `langchain.tools.tool` | Re-export of `@tool` decorator (same as `langchain_core.tools.tool`). |
| `langchain.embeddings.init_embeddings` | Provider-neutral embeddings factory. |

### `langchain-google-genai` (`langchain_google_genai.*`)

| Import | What it is |
|---|---|
| `langchain_google_genai.ChatGoogleGenerativeAI` | Gemini chat model. |
| `langchain_google_genai.GoogleGenerativeAIEmbeddings` | Gemini embeddings. |
| `langchain_google_genai.GoogleGenerativeAI` | **Legacy text-completion model. Don't use in new code.** |
| `langchain_google_genai.HarmCategory`, `HarmBlockThreshold` | Safety settings enums. |

### `langgraph` (`langgraph.*`)

| Import | What it is |
|---|---|
| `langgraph.graph.{StateGraph, START, END, MessagesState}` | The graph builder + sentinels + prebuilt messages state. |
| `langgraph.graph.message.add_messages` | Reducer for message lists. |
| `langgraph.types.{Send, Command, Interrupt, interrupt, StreamWriter}` | Control-flow primitives. |
| `langgraph.runtime.Runtime` | Per-invocation runtime context (replaces v0.x `config_schema`). |
| `langgraph.config.{get_stream_writer, get_config, get_store}` | Helpers to access runtime context inside nodes. |
| `langgraph.prebuilt.{create_react_agent, ToolNode, tools_condition, InjectedState, InjectedStore}` | Prebuilt agent + tool node. |
| `langgraph.checkpoint.base.BaseCheckpointSaver` | Base class. |
| `langgraph.checkpoint.memory.InMemorySaver` | In-memory checkpointer. |
| `langgraph.checkpoint.sqlite.SqliteSaver` / `langgraph.checkpoint.sqlite.aio.AsyncSqliteSaver` | SQLite checkpointer. |
| `langgraph.checkpoint.postgres.PostgresSaver` / `langgraph.checkpoint.postgres.aio.AsyncPostgresSaver` | Postgres checkpointer. |
| `langgraph.checkpoint.serde.encrypted.EncryptedSerializer` | Checkpoint encryption. |
| `langgraph.store.base.BaseStore` | Base class for long-term stores. |
| `langgraph.store.memory.InMemoryStore` | In-memory store with optional embeddings. |
| `langgraph.store.postgres.PostgresStore` | Postgres-backed store. |
| `langgraph.errors.{GraphRecursionError, NodeInterrupt, GraphInterrupt, InvalidUpdateError}` | Exception types. |

### `langsmith` (`langsmith.*`)

| Import | What it is |
|---|---|
| `langsmith.{Client, traceable, trace}` | Core entry points. |
| `langsmith.{get_current_run_tree, tracing_context}` | Run-context helpers. |
| `langsmith.run_trees.RunTree` | Programmatic run construction. |
| `langsmith.wrappers.{wrap_openai, wrap_anthropic}` | Auto-trace third-party SDKs. |
| `langsmith.{evaluate, aevaluate}` | Run an evaluation experiment. |
| `langsmith.evaluation.{LangChainStringEvaluator}` | Built-in evaluator wrappers. |
| `langsmith.schemas.{Run, Example, Dataset, Feedback}` | Pydantic types for SDK responses. |

## What changed in the v1.0 transition (Oct 2025)

The v1 release (and follow-ups) reshaped the API. Here's what's different and where the old stuff went:

### Moved to `langchain-classic` (do not use in new code)

These imports still work *if* you `pip install langchain-classic` and update the import path, but they should be migrated:

| Old (pre-v1) | What to use now |
|---|---|
| `langchain.chains.LLMChain` | LCEL: `prompt | llm | parser` |
| `langchain.chains.SequentialChain` / `SimpleSequentialChain` | LCEL piping |
| `langchain.chains.ConversationalRetrievalChain` | A LangGraph graph (`StateGraph`) with a retrieval node + chat node |
| `langchain.chains.RetrievalQA` / `RetrievalQAWithSourcesChain` | LCEL RAG chain (see `08-rag-pipeline.md`) |
| `langchain.agents.initialize_agent` | `langchain.agents.create_agent` |
| `langchain.agents.AgentExecutor` | `langchain.agents.create_agent` returns a compiled graph that replaces this |
| `langchain.memory.ConversationBufferMemory` | LangGraph checkpointer + `MessagesState` |
| `langchain.memory.ConversationBufferWindowMemory` | LangGraph + custom message-trimming logic |
| `langchain.memory.ConversationSummaryMemory` | `SummarizationMiddleware` (in `langchain.agents.middleware`) |
| `langchain.memory.ConversationSummaryBufferMemory` | `SummarizationMiddleware` |
| `langchain.memory.VectorStoreRetrieverMemory` | LangGraph `Store` with semantic search |
| `langchain.memory.ConversationKGMemory` | Custom store + extraction logic |

### Behavioral changes that affect existing code

1. **`create_agent`'s `state_schema` must be a `TypedDict`.** Pydantic / dataclass agent state is no longer accepted (LangGraph `StateGraph` itself still accepts all three for `state_schema`).
2. **Standard content blocks.** Messages now expose a `.content_blocks` property that normalizes multimodal/reasoning content across providers. The `.content` field is unchanged for backward compatibility.
3. **Tool calling normalized across providers.** `AIMessage.tool_calls` is the canonical form: `[{"name": ..., "args": ..., "id": ...}]`. Provider-specific shapes are translated under the hood.
4. **Streaming events default version is `"v2"`.** Old `version="v1"` is still accepted but discouraged.
5. **`config_schema` parameter on `StateGraph` was renamed to `context_schema`** (in LangGraph v0.6, fully removed in v2.0). Use `context_schema` and access via `Runtime.context` inside nodes.
6. **`thinking_budget` deprecated for Gemini 3+.** Use `thinking_level` instead. (Gemini 2.5 still uses `thinking_budget`.)

### Things that look deprecated but actually still work

- **`langgraph.prebuilt.create_react_agent`** — still fully supported. It's just lower-level than `langchain.agents.create_agent`. Use it when you want graph-level customization (extra nodes, custom routing) without the middleware framework.
- **`RunnableWithMessageHistory`** — still supported for simple per-session history when you don't want to bring in LangGraph. For anything richer, migrate to LangGraph.
- **`PydanticOutputParser`** — still works, but `model.with_structured_output(Schema)` is dramatically more reliable on Gemini and other tool-calling models. Reserve `PydanticOutputParser` for models that don't support structured output natively.

## Provider neutrality (when to use `init_chat_model`)

`langchain.chat_models.init_chat_model` lets you select providers via a string identifier:

```python
from langchain.chat_models import init_chat_model

llm = init_chat_model("google_genai:gemini-2.5-flash", temperature=0.2)
# Equivalent providers: "openai:gpt-5", "anthropic:claude-sonnet-4-5",
#                      "google_vertexai:gemini-2.5-pro", "ollama:llama3.1"
```

Use this when:
- You want the model selectable from config / env / spec.
- You're writing reusable components that shouldn't hardcode a provider.

Use the explicit class (`ChatGoogleGenerativeAI(...)`) when:
- You need provider-specific parameters (e.g., `thinking_level`, `safety_settings`).
- You want type-checker-visible parameter names.

For new code targeting Gemini specifically, prefer the explicit class so all Gemini-specific options are accessible.

## Quick decision rules

- **Building an agent?** → `langchain.agents.create_agent`.
- **Building a non-agent chain?** → LCEL piping with `|`.
- **Need persistence/memory?** → LangGraph `StateGraph` + checkpointer (and `Store` for cross-thread).
- **Need structured output?** → `model.with_structured_output(Schema)`.
- **Need tracing?** → Set `LANGSMITH_TRACING=true` env var; nothing else needed.
- **Need per-session message history without LangGraph?** → `RunnableWithMessageHistory`.
- **Need to fan out parallel work?** → LangGraph `Send` API or LCEL `RunnableParallel`.
- **Need human approval before tool execution?** → `HumanInTheLoopMiddleware`.

For deeper detail on any of these, consult the specific reference files listed in `SKILL.md`.