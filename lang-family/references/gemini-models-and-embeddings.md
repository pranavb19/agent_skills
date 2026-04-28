# 01 — Gemini Models and Embeddings (`langchain-google-genai`)

## Package: `langchain-google-genai` 4.x

Uses the consolidated `google-genai` SDK internally. Supports both the **Gemini Developer API** (api key) and **Vertex AI** (GCP project + ADC) as backends.

---

## `ChatGoogleGenerativeAI` — chat model

### Import

```python
from langchain_google_genai import ChatGoogleGenerativeAI
```

### Authentication

**Developer API (default):**
```python
import os
os.environ["GOOGLE_API_KEY"] = "AIza..."      # or GEMINI_API_KEY (both recognized)

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
# OR pass directly:
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key="AIza...")
```

**Vertex AI:**
```python
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "true"
os.environ["GOOGLE_CLOUD_PROJECT"] = "my-project"
os.environ["GOOGLE_CLOUD_LOCATION"] = "us-central1"   # optional, defaults to us-central1
# Application Default Credentials (ADC) used: gcloud auth application-default login

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
# OR constructor kwargs:
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    vertexai=True,
    project="my-gcp-project",
    location="us-central1",
)
```

### Constructor parameters (all optional except `model`)

```python
llm = ChatGoogleGenerativeAI(
    # ── Identity ────────────────────────────────────────────────────
    model="gemini-2.5-flash",       # str: required. See model table below.

    # ── Generation ───────────────────────────────────────────────────
    temperature=0.2,                # float [0, 2]. IMPORTANT: for Gemini 3+, SDK
                                    # overrides to 1.0 if you set <1.0 — this is
                                    # intentional to prevent loops. Leave unset for
                                    # Gemini 3 unless you have a specific reason.
    top_p=0.95,                     # float nucleus sampling
    top_k=40,                       # int top-k sampling
    max_output_tokens=2048,         # int: cap generation length
    candidate_count=1,              # int: number of completions to generate
    stop_sequences=["##END##"],     # list[str]: stop at these strings

    # ── Reasoning (model-specific) ───────────────────────────────────
    # Gemini 3+ (gemini-3-*):
    thinking_level="medium",        # "minimal" | "low" | "medium" | "high" (default "high")
                                    # Controls how much internal reasoning the model does.
    # Gemini 2.5 (gemini-2.5-*):
    thinking_budget=1024,           # int: tokens for internal reasoning.
                                    # 0 = disable, -1 = dynamic, positive int = fixed budget.
                                    # DEPRECATED for Gemini 3+ — use thinking_level.

    # ── Safety ───────────────────────────────────────────────────────
    safety_settings={               # dict or list of SafetySetting objects
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    },

    # ── Response format ──────────────────────────────────────────────
    response_mime_type="text/plain",  # "text/plain" | "application/json"
                                      # Usually controlled via with_structured_output() instead.
    response_schema=None,            # JSON schema dict; use with_structured_output() instead.

    # ── Context caching (Gemini 2.5+) ────────────────────────────────
    cached_content=None,             # str: name of a cached content resource from the Files API

    # ── Reliability ──────────────────────────────────────────────────
    max_retries=2,                   # int: automatic retry on transient errors
    timeout=60.0,                    # float: per-request timeout in seconds
    request_timeout=None,            # alias for timeout

    # ── Vertex AI options ────────────────────────────────────────────
    vertexai=False,                  # bool: use Vertex AI backend
    project=None,                    # str: GCP project ID (Vertex only)
    location="us-central1",          # str: GCP region (Vertex only)
    credentials=None,                # google.auth.credentials.Credentials object

    # ── Misc ─────────────────────────────────────────────────────────
    client_options=None,             # google.api_core.client_options.ClientOptions
    transport=None,                  # "rest" | "grpc" | None (auto-detect)
    n=1,                             # alias for candidate_count
    streaming=False,                 # bool: prefer streaming internally (use .stream() instead)
    convert_system_message_to_human=False,  # Legacy: was needed for old Gemini models
                                            # that didn't support system prompts natively.
                                            # Not needed for Gemini 2.5/3+.
)
```

### Supported models (April 2026)

| Model ID | Context window | Best for | Notes |
|---|---|---|---|
| `gemini-2.5-flash` | 1M tokens | **Default choice.** Fast, cheap, very capable. | Stable |
| `gemini-2.5-pro` | 1M tokens | Hard reasoning, long context. | Stable |
| `gemini-2.5-flash-lite` | 1M tokens | High-volume, cost-sensitive. | Stable |
| `gemini-3-flash-preview` | 1M tokens | Fastest Gemini 3 generation. | Preview |
| `gemini-3-pro-preview` | 2M tokens | Most capable, full reasoning. | Preview |
| `gemini-embedding-001` | N/A | Embeddings — corpus + query. | Stable |
| `gemini-embedding-2-preview` | N/A | Embeddings next-gen. | Preview |

**Deprecated (shut down June 1, 2026):** `gemini-2.0-flash`, `gemini-2.0-pro`, `gemini-2.0-flash-lite`, and all `gemini-1.5-*` models. Do not use in new code.

For the current canonical list: `https://ai.google.dev/gemini-api/docs/models`

### Basic invocation

```python
from langchain_core.messages import HumanMessage, SystemMessage

# String shorthand (becomes HumanMessage)
response = llm.invoke("Explain transformers in one sentence.")
print(response.content)  # str

# List of messages (preferred)
response = llm.invoke([
    SystemMessage(content="You are a concise technical writer."),
    HumanMessage(content="Explain transformers."),
])

# Tuple shorthand (role, content) — identical to message objects
response = llm.invoke([
    ("system", "You are a concise technical writer."),
    ("human", "Explain transformers."),
])
```

### Token counting

```python
# Count tokens before sending (returns int)
n = llm.get_num_tokens("Your text here")
n = llm.get_num_tokens_from_messages(messages)  # list of messages
```

Use this before large context calls to verify you're within the model's window.

### Multimodal inputs

Pass a list in `HumanMessage.content` where each element is a dict with a `type` key.

**Inline image (base64, best for one-off small images):**
```python
import base64
from langchain_core.messages import HumanMessage

with open("chart.png", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

msg = HumanMessage(content=[
    {"type": "text", "text": "Describe this chart."},
    {"type": "image_url", "image_url": f"data:image/png;base64,{b64}"},
])
response = llm.invoke([msg])
```

**Image by public URL:**
```python
msg = HumanMessage(content=[
    {"type": "text", "text": "What's in this image?"},
    {"type": "image_url", "image_url": "https://example.com/photo.jpg"},
])
```

**Audio/Video/PDF (inline base64):**
```python
msg = HumanMessage(content=[
    {"type": "text", "text": "Transcribe this audio."},
    {"type": "media", "data": audio_b64, "mime_type": "audio/mpeg"},
])
# Other mime_types: "video/mp4", "application/pdf", "audio/wav", "image/jpeg", etc.
```

**File API (recommended for files > 20 MB or files reused across requests):**
```python
import google.genai as genai

genai_client = genai.Client(api_key="...")
uploaded = genai_client.files.upload(path="lecture.mp4")  # returns a File object

msg = HumanMessage(content=[
    {"type": "text", "text": "Summarize this lecture."},
    {"type": "media", "file_uri": uploaded.uri, "mime_type": "video/mp4"},
])
# OR reference by file_id if the SDK exposes it
```

### Safety settings

```python
from langchain_google_genai import HarmCategory, HarmBlockThreshold

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    safety_settings={
        HarmCategory.HARM_CATEGORY_HARASSMENT:     HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH:    HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    },
)
```

Thresholds: `BLOCK_NONE`, `BLOCK_ONLY_HIGH`, `BLOCK_MEDIUM_AND_ABOVE`, `BLOCK_LOW_AND_ABOVE`.

### Grounding with Google Search (Gemini 2.5+)

```python
from google.genai.types import Tool, GoogleSearch

llm_grounded = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    tools=[Tool(google_search=GoogleSearch())],
)
# Use this when you want the model to search the web during generation.
# Not for agent tool-calling — this is a model-level capability.
```

### `with_structured_output` (the canonical structured output path)

Always prefer this over `PydanticOutputParser` for Gemini.

```python
from pydantic import BaseModel, Field

class Movie(BaseModel):
    title: str = Field(description="Movie title")
    year: int = Field(description="Release year")
    genre: str = Field(description="Primary genre")

# method="json_schema" uses Gemini's native response_json_schema (default, most reliable)
structured_llm = llm.with_structured_output(Movie, method="json_schema")
result = structured_llm.invoke("Tell me about Inception.")
# result is a Movie instance

# method="function_calling" uses tool-calling (fallback if json_schema not available)
structured_llm_fc = llm.with_structured_output(Movie, method="function_calling")

# include_raw=True returns both raw AIMessage and parsed result:
structured_llm_raw = llm.with_structured_output(Movie, include_raw=True)
out = structured_llm_raw.invoke("Tell me about Inception.")
# out = {"raw": AIMessage(...), "parsed": Movie(...), "parsing_error": None}
```

**What Gemini's `json_schema` method does:** the model is constrained by the SDK to only produce valid JSON matching your schema. This is dramatically more reliable than prompt-injected format instructions + PydanticOutputParser because:
- No parse errors from model hallucinating invalid JSON.
- No need for format instructions in the prompt.
- Works for complex nested schemas.

**Schema compatibility notes:**
- `Union[X, None]` / `Optional[X]` is supported (maps to nullable in the schema).
- `$defs` are auto-inlined by the SDK.
- `Literal[...]` types work for enum-like constraints.
- Very deep nesting (>5 levels) may cause issues — flatten if possible.
- You can also pass a plain `dict` (JSON schema format) or `TypedDict` instead of Pydantic.

### Binding tools (for agents/tool-calling)

```python
from langchain_core.tools import tool

@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"

llm_with_tools = llm.bind_tools([search])
ai_msg = llm_with_tools.invoke("Find info about LangChain.")
print(ai_msg.tool_calls)
# [{"name": "search", "args": {"query": "LangChain"}, "id": "call_abc123"}]
```

In practice you don't call `bind_tools` directly in agent code — `create_agent` and `create_react_agent` do it internally.

---

## `GoogleGenerativeAIEmbeddings` — embedding model

### Import

```python
from langchain_google_genai import GoogleGenerativeAIEmbeddings
```

### Critical: `task_type` parameter

This is the single most important parameter and is almost always set wrong in tutorials. Gemini's embedding model generates different vector representations depending on the intended use — the embeddings are **optimized for different tasks and live in different aligned sub-spaces**.

| `task_type` | When to use |
|---|---|
| `"RETRIEVAL_DOCUMENT"` | Embedding the **corpus** (chunks being stored in the vector DB). |
| `"RETRIEVAL_QUERY"` | Embedding the **search query** (user's question). |
| `"SEMANTIC_SIMILARITY"` | General similarity comparisons. Use for the `InMemoryStore` index when doing memory search. |
| `"CLASSIFICATION"` | Text classification tasks. |
| `"CLUSTERING"` | Grouping documents. |
| `"QUESTION_ANSWERING"` | When the input text is the question and you're querying stored answers. |
| `"FACT_VERIFICATION"` | Fact-checking tasks. |
| `"CODE_RETRIEVAL_QUERY"` | Embedding a natural-language query to retrieve code. |

**RAG gotcha:** use `RETRIEVAL_DOCUMENT` when calling `embed_documents` (indexing) and `RETRIEVAL_QUERY` when calling `embed_query` (at query time). The two embeddings are aligned asymmetrically — they are NOT interchangeable. If you use the same task_type for both, similarity search will silently underperform.

**Practical pattern for RAG: two separate embedding objects:**

```python
doc_embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    task_type="RETRIEVAL_DOCUMENT",
)
query_embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    task_type="RETRIEVAL_QUERY",
)

# Index time: use doc_embeddings
vector_store = Chroma(embedding_function=doc_embeddings)
vector_store.add_documents(chunks)

# Query time: pass query_embeddings to the retriever
retriever = vector_store.as_retriever(
    search_kwargs={"k": 5},
    # Override the embedding function at query time:
    # vector_store._embedding_function = query_embeddings  ← some stores support this
)
# Simplest: just use vector_store.similarity_search() with the right embedder
results = vector_store.similarity_search_with_score(
    query,
    k=5,
    # The base vector store will use doc_embeddings for this — to override:
)
# Better pattern: create the vector store with doc_embeddings, then call embed_query
# explicitly to compute the query vector and pass it to search_by_vector:
query_vec = query_embeddings.embed_query("What is LangGraph?")
results = vector_store.similarity_search_by_vector(query_vec, k=5)
```

### Constructor

```python
emb = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",    # or "gemini-embedding-2-preview"
    task_type="RETRIEVAL_DOCUMENT",  # see table above
    output_dimensionality=768,       # Optional: MRL-based dimension reduction.
                                     # Supported values: 768, 1536, 3072 (model-dependent).
                                     # Lower dims = faster + smaller index, slightly lower quality.
    api_key="...",                   # or GOOGLE_API_KEY env var
    # Vertex AI:
    vertexai=False,
    project=None,
    location="us-central1",
    credentials=None,
    # Reliability:
    max_retries=2,
    timeout=60.0,
)
```

### Methods

```python
# Embed a list of documents (for indexing)
vectors: list[list[float]] = emb.embed_documents(["doc1", "doc2", ...])
# Max batch size: 100 strings. SDK auto-batches if you pass more.

# Embed a single query string
vector: list[float] = emb.embed_query("What is X?")

# Async variants
vectors = await emb.aembed_documents(["doc1", "doc2"])
vector  = await emb.aembed_query("What is X?")
```

### Embedding dimensions

`gemini-embedding-001` produces 3072-dimensional vectors by default. You can reduce to 768 or 1536 via `output_dimensionality` without significant quality loss for most tasks (MRL = Matryoshka Representation Learning).

Match the dimension you configure here to your vector store's expected dimension.

```python
# If using FAISS, create the index with the right dimension:
import faiss
dim = len(emb.embed_query("test"))  # 3072 by default, or 768 if output_dimensionality=768
index = faiss.IndexFlatL2(dim)
```

### For `InMemoryStore` / `PostgresStore` semantic memory

Use `SEMANTIC_SIMILARITY` task type for cross-thread memory stores (not retrieval):

```python
from langchain_google_genai import GoogleGenerativeAIEmbeddings

store_emb = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    task_type="SEMANTIC_SIMILARITY",
    output_dimensionality=768,       # reduce for performance in the store
)
```

---

## Legacy: `GoogleGenerativeAI` (text-completion)

This class is for text completion (non-chat) models. It is **explicitly marked legacy** in the package reference and should not be used. Use `ChatGoogleGenerativeAI` for all new code. The Gemini API no longer has meaningful completion-only models.

```python
# DO NOT USE:
from langchain_google_genai import GoogleGenerativeAI   # legacy, avoid
```

---

## Gotchas and best practices

1. **Temperature and Gemini 3+:** The SDK overrides `temperature` to 1.0 for `gemini-3-*` models if you set it lower. This is intentional — lower temperatures cause loop/degradation issues in the Gemini 3 generation. Don't fight it; use `thinking_level` to control the quality/cost tradeoff instead.

2. **`thinking_level` vs `thinking_budget`:** `thinking_level` is for Gemini 3+; `thinking_budget` is for Gemini 2.5. Using `thinking_budget` on a Gemini 3 model raises a warning and is ignored. Check your model string.

3. **`convert_system_message_to_human=False`:** This was needed for old Gemini 1.0 Pro models that didn't support system messages. It's NOT needed for Gemini 2.5+ and should not be set unless targeting legacy models.

4. **Streaming and `with_structured_output`:** Native JSON schema mode (`method="json_schema"`) doesn't stream partial structured objects — you get the full response at the end. If you need streaming + structure, use `method="function_calling"` or post-process streamed JSON yourself.

5. **Parallel requests:** `batch()` and `abatch()` issue requests concurrently. With Gemini, rate limits are per-project per model; tune `max_concurrency` in `RunnableConfig` accordingly.

6. **Vertex AI vs Developer API:** Vertex AI has higher quotas, regional data residency, and enterprise SLAs. Developer API (api key) is simpler for prototyping. The model IDs are the same; only auth differs.

7. **Model name aliases:** Some tutorials use `"gemini-pro"` or `"gemini-1.0-pro"`. These are deprecated. Always use the full versioned name (`"gemini-2.5-flash"`, etc.).
