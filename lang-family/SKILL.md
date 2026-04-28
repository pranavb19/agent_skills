---
name: lang-family-skill
description: Master reference for building production AI agents with the LangChain Python ecosystem (LangChain 1.3+, LangGraph 1.1+, LangSmith 0.7+) using Google Gemini via langchain-google-genai. Use this skill whenever the user wants to build, scaffold, modify, or debug code that uses LangChain, LangGraph, LangSmith, ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings, create_agent, StateGraph, checkpointers, tool-calling agents, RAG pipelines, multi-agent systems, agent middleware, structured output, streaming, or LLM-app tracing/observability — even when they don't explicitly name the libraries (e.g., "build me a Gemini agent that reads PDFs", "add memory to my chatbot", "make this agent pause for human approval", "trace my agent calls", "set up evaluation for my LLM app"). Trigger eagerly: any agentic Python code involving Gemini, tool calls, retrieval, conversation memory, persistence, or LLM observability is in scope.
---

# LangChain / LangGraph / LangSmith Master Skill (Gemini-focused, April 2026)

A complete, code-focused reference for building AI agents end-to-end with the LangChain Python SDK family. This skill targets the **stable v1.x line**: LangChain 1.3.x, LangGraph 1.1.x, LangSmith 0.7.x, `langchain-google-genai` 4.x. Code defaults to **Google Gemini** via `langchain-google-genai`.

This SKILL.md is a router. The actual depth lives in `references/`. Read the relevant reference files when generating or modifying code — don't rely on memory for API surface details, because the v1.0 transition (Oct 2025) deprecated a lot of pre-v1 patterns.

---

## When to use this skill

Use it whenever the task involves any of:

- Building a Gemini-powered agent (single-agent, multi-agent, ReAct, supervisor, swarm)
- Wiring up `ChatGoogleGenerativeAI` or `GoogleGenerativeAIEmbeddings`
- Defining tools (`@tool`, `StructuredTool`, `BaseTool`) and binding them to a model
- Composing chains with LCEL (`|`, `RunnableParallel`, `RunnableLambda`, `RunnablePassthrough`)
- Building a RAG pipeline (loaders → splitters → embeddings → vector stores → retrievers)
- Building a `StateGraph` (state schemas, reducers, conditional edges, `Send`, `Command`)
- Adding persistence (`InMemorySaver`, `SqliteSaver`, `PostgresSaver`) or long-term memory (`Store`)
- Adding human-in-the-loop interrupts (`interrupt()`, `Command(resume=...)`)
- Adding LangSmith tracing, OpenTelemetry export, evaluation, or feedback loops
- Adding agent middleware (summarization, HITL, model fallback, PII, tool retry, etc.)
- Streaming tokens or graph state, or running things async/batch
- Anything calling itself "an agent" or "an LLM app" in Python

If the user mentions Gemini, agents, RAG, vector stores, tool calling, conversation memory, durable workflows, agent observability, or LLM evaluation — assume this skill is in scope and read the relevant references before writing code.

---

## Hard rules (the v1 transition — these are easy to get wrong from training data)

These are decisions that should be made the same way every time. They override anything that contradicts them in older tutorials, blog posts, or model memory.

1. **Use `langchain.agents.create_agent` for new agents**, not `initialize_agent`, not `AgentExecutor`, not `LLMChain`. The latter three are moved to the separate compatibility package `langchain-classic` and should never appear in new code.
2. **Use LCEL piping (`prompt | llm | parser`) for chains**, not `LLMChain` / `SequentialChain` / `ConversationalRetrievalChain` (also `langchain-classic`).
3. **Use LangGraph checkpointers + stores for memory**, not `ConversationBufferMemory` / `ConversationSummaryMemory` / `ConversationBufferWindowMemory` (also `langchain-classic`). For very simple per-session message history without LangGraph, `RunnableWithMessageHistory` still works.
4. **Use `with_structured_output(Schema, method="json_schema")` for Gemini structured output**, not `PydanticOutputParser` glued onto a prompt. The native Gemini `response_json_schema` path is dramatically more reliable.
5. **Custom `state_schema` on `create_agent` MUST be a `TypedDict`** in v1 (Pydantic and dataclasses no longer accepted there). LangGraph `StateGraph` itself still accepts `TypedDict`, Pydantic `BaseModel`, and `dataclass`.
6. **Always use a reducer (`Annotated[T, reducer]`) on any state channel that is updated by more than one node, by `Send`, or by `Command(graph=Command.PARENT)`.** For messages: `Annotated[list[AnyMessage], add_messages]` or just inherit `MessagesState`.
7. **Always pass `config={"configurable": {"thread_id": "..."}}`** when invoking a graph that has a checkpointer. No thread_id = no persistence = errors.
8. **For Gemini 3+**: leave `temperature` at default (1.0). The `langchain-google-genai` SDK overrides lower temperatures to 1.0 for Gemini 3 because lower values cause loops/degraded output.
9. **For Gemini reasoning controls**: Gemini 3+ uses `thinking_level` (`"minimal" | "low" | "medium" | "high"`); Gemini 2.5 uses `thinking_budget` (int tokens, 0=off, -1=dynamic). Don't mix them.
10. **For Gemini embeddings, set `task_type` correctly**: `"RETRIEVAL_DOCUMENT"` when embedding the corpus, `"RETRIEVAL_QUERY"` when embedding incoming queries. Index and query embeddings live in different aligned spaces — getting this wrong silently destroys retrieval quality.
11. **`create_react_agent` exists in two places**: `langchain.agents.create_agent` (preferred, middleware-based, in `langchain` package) and `langgraph.prebuilt.create_react_agent` (lower-level, still supported, useful when you want graph-level customization without middleware). Don't confuse them.
12. **Python ≥ 3.10** is required across the stack.

---

## Reference index

When the user's task touches one of these areas, read the listed file(s) BEFORE writing code. Most non-trivial tasks need 2–4 references combined.

### Foundations
- **`references/00-ecosystem-and-versions.md`** — Package layout, what's in `langchain` vs `langchain-core` vs `langchain-community` vs `langgraph` vs `langsmith` vs `langchain-google-genai`. v1.0 transition changes. What's deprecated and where it moved.
- **`references/01-gemini-models-and-embeddings.md`** — `ChatGoogleGenerativeAI` constructor, all parameters, auth (API key vs Vertex AI), available models, multimodal input formats, `GoogleGenerativeAIEmbeddings` and the critical `task_type` parameter.
- **`references/02-prompts-and-templates.md`** — `PromptTemplate`, `ChatPromptTemplate`, `MessagesPlaceholder`, `FewShotPromptTemplate`, `FewShotChatMessagePromptTemplate`, example selectors.
- **`references/03-output-parsers-and-structured-output.md`** — `StrOutputParser`, `JsonOutputParser`, `PydanticOutputParser`, `XMLOutputParser`. Strongly preferred path: `model.with_structured_output(Schema)`.

### Composition
- **`references/04-lcel-and-runnables.md`** — The `Runnable` interface, pipe operator, `RunnableSequence`, `RunnableParallel`, `RunnableLambda`, `RunnablePassthrough`, `RunnableBranch`, `RunnableConfig`, `.with_retry()`, `.with_fallbacks()`, `.assign()`, `.bind()`.
- **`references/05-tools.md`** — `@tool` decorator, `StructuredTool`, `BaseTool` subclassing, `args_schema`, tool calling with Gemini, `bind_tools`.
- **`references/06-agents-create-agent.md`** — `langchain.agents.create_agent` complete API, agent state, `response_format` (ProviderStrategy vs ToolStrategy), invocation patterns.
- **`references/07-middleware.md`** — All middleware hooks (`before_model`, `after_model`, `wrap_model_call`, etc.), built-in middleware classes (`SummarizationMiddleware`, `HumanInTheLoopMiddleware`, `ModelCallLimitMiddleware`, `ToolCallLimitMiddleware`, `ModelFallbackMiddleware`, `PIIMiddleware`, etc.), writing custom middleware.

### Retrieval-Augmented Generation
- **`references/08-rag-pipeline.md`** — Document loaders, text splitters, embeddings, vector stores (FAISS, Chroma, Pinecone, Weaviate), retrievers (`MultiQueryRetriever`, `EnsembleRetriever`, `ContextualCompressionRetriever`, `ParentDocumentRetriever`, `SelfQueryRetriever`), reranking, complete RAG chain construction. This is a long file — has its own table of contents.

### Runtime
- **`references/09-streaming-and-async.md`** — `.stream`, `.astream`, `.astream_events(version="v2")`, `.batch`, `.abatch`, custom events via `adispatch_custom_event`, `disable_streaming` flag.
- **`references/10-caching.md`** — `set_llm_cache`, `InMemoryCache`, `SQLiteCache`, `RedisCache`, `RedisSemanticCache`.
- **`references/11-memory.md`** — `RunnableWithMessageHistory`, `BaseChatMessageHistory` implementations. Note: for non-trivial memory use LangGraph (file 17).

### LangGraph
- **`references/12-langgraph-stategraph.md`** — `StateGraph` constructor, state schemas (TypedDict / Pydantic / dataclass), reducers (`Annotated`, `add_messages`, `operator.add`), nodes, edges, `START` / `END`, `compile()`, the `Runtime` API.
- **`references/13-langgraph-control-flow.md`** — `add_conditional_edges`, the `Send` API for fanout/map-reduce, the `Command` primitive (state update + routing in one return), `Command.PARENT` for cross-graph hops.
- **`references/14-langgraph-interrupts-hitl.md`** — `interrupt()`, `Command(resume=...)`, static `interrupt_before`/`interrupt_after`, multi-interrupt indexing rules, the re-execution semantic on resume.
- **`references/15-langgraph-checkpointers.md`** — `BaseCheckpointSaver` interface, `InMemorySaver`, `SqliteSaver`, `AsyncSqliteSaver`, `PostgresSaver`, `AsyncPostgresSaver`, encryption (`EncryptedSerializer`), thread_id / checkpoint_id / checkpoint_ns.
- **`references/16-langgraph-time-travel.md`** — `get_state`, `get_state_history`, `update_state`, replay vs fork semantics, `as_node` parameter.
- **`references/17-langgraph-stores.md`** — `BaseStore`, `InMemoryStore`, `PostgresStore`, namespaces, semantic search via embeddings, the `langmem` library helpers.
- **`references/18-langgraph-prebuilt-agents.md`** — `langgraph.prebuilt.create_react_agent`, `ToolNode`, `tools_condition`, when to use prebuilt vs `langchain.agents.create_agent` vs raw `StateGraph`.
- **`references/19-langgraph-multi-agent.md`** — Supervisor (`langgraph-supervisor`), Swarm (`langgraph-swarm`), hierarchical/subgraphs, agent-as-tool, handoff patterns.
- **`references/20-langgraph-streaming.md`** — `stream_mode` values (`"values"`, `"updates"`, `"messages"`, `"custom"`, `"checkpoints"`, `"tasks"`, `"debug"`), `version="v2"` unified `StreamPart` shape, `get_stream_writer` for custom events.
- **`references/21-langgraph-advanced-patterns.md`** — Plan-and-execute, reflection / self-critique, tree-of-thoughts, reflexion, agent-as-tool, dynamic graph construction.

### LangSmith
- **`references/22-langsmith-tracing.md`** — Setup env vars, project config, automatic tracing for LangChain/LangGraph, run types, run tree structure, metadata/tags, distributed tracing across services.
- **`references/23-langsmith-traceable.md`** — `@traceable` decorator deep dive, `langsmith_extra`, `wrappers.wrap_openai` / `wrap_anthropic`, `get_current_run_tree`, `tracing_context`, sync/async/generator support.
- **`references/24-langsmith-opentelemetry.md`** — OTel modes (LangSmith → OTel, OTel → LangSmith, global tracer provider), `LANGSMITH_OTEL_ENABLED`, OTLP endpoints, semantic conventions, `langsmith.span.kind` and friends. **The user's most-requested LangSmith topic.**
- **`references/25-langsmith-client-api.md`** — `Client` class methods: `create_dataset`, `create_examples`, `list_runs`, `create_feedback` (and the critical `trace_id=` for batched ingestion).
- **`references/26-langsmith-evaluation-datasets.md`** — `evaluate`, `aevaluate`, evaluator types (heuristic, LLM-as-judge, pairwise, summary), datasets and versioning, the pytest plugin.
- **`references/27-langsmith-monitoring-feedback.md`** — Auto-tracked metrics (latency, tokens, cost, errors), online evaluators, custom dashboards, production feedback loops.

### Cross-cutting
- **`references/28-best-practices-and-gotchas.md`** — Versioning gotchas, Gemini-specific quirks, LangGraph correctness rules, LangSmith hygiene, performance tips, safety patterns. Always skim this when wrapping up a non-trivial implementation.
- **`references/29-end-to-end-example.md`** — Complete idiomatic Gemini agent: tools + structured output + Postgres persistence + Postgres store with semantic memory + tracing + middleware + streaming. Use as a template/cookbook for production scaffolding.

---

## Working with this skill

### When generating code from a spec

1. **Identify the surface area.** Which references apply? A "Gemini agent that does RAG over PDFs and remembers conversation history" needs: `01` (Gemini), `06` (create_agent), `08` (RAG), `15` (checkpointers), and probably `07` (middleware for summarization).

2. **Read those references first.** Do not write code from memory — the v1 changes are too easy to get wrong.

3. **Use the canonical patterns from the references.** They're version-correct as of the v1.x stable line.

4. **Cross-check against `28-best-practices-and-gotchas.md`** before delivering. Common things to verify: TypedDict for `state_schema` on `create_agent`, reducers on shared state channels, `thread_id` in invocation config, correct `task_type` on Gemini embeddings.

5. **Use `29-end-to-end-example.md` as scaffolding** for full applications — it's organized so each piece can be lifted independently.

### When the user asks "how do I X?"

Give a short answer in chat for simple questions. For anything beyond a few lines, generate working code. Quote API surface details from the references rather than recalling them — they're authoritative for this stack as of April 2026.

### When the user's existing code uses pre-v1 patterns

If you encounter `LLMChain`, `initialize_agent`, `AgentExecutor`, `ConversationBufferMemory`, etc., flag the migration path and offer to rewrite. Cite the relevant reference (`00-ecosystem-and-versions.md` covers what moved where).

### Style for generated code

- Default to **`ChatGoogleGenerativeAI(model="gemini-2.5-flash")`** unless the user asks for something else. Use `"gemini-2.5-pro"` for hard reasoning, `"gemini-2.5-flash-lite"` for cheap/fast.
- Default to **`langchain.agents.create_agent`** for agents.
- Default to **`with_structured_output`** for any structured response.
- Default to **`PostgresSaver`** for production persistence, **`SqliteSaver`** for local dev, **`InMemorySaver`** only for one-shot examples/tests.
- Default to **LCEL piping (`|`)** for non-agent chains.
- Always include explicit imports, not wildcard.
- Use type hints on every function (this is what tools' `args_schema` is auto-derived from).
- For tool functions, write a precise docstring — Gemini sees the docstring as the tool description.
- When introducing a checkpointer, always add a thread_id example in the same code block.