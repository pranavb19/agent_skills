# 08 — RAG Pipeline (Complete Guide)

## Table of Contents

1. [Document Loaders](#1-document-loaders)
2. [Text Splitters](#2-text-splitters)
3. [Embeddings](#3-embeddings) (→ see `01-gemini-models-and-embeddings.md`)
4. [Vector Stores](#4-vector-stores)
5. [Retrievers](#5-retrievers)
6. [Reranking](#6-reranking)
7. [Complete RAG Chain (LCEL)](#7-complete-rag-chain-lcel)
8. [Conversational RAG](#8-conversational-rag-as-langgraph)
9. [Advanced Patterns](#9-advanced-patterns)
10. [Gotchas](#10-gotchas)

---

## 1. Document Loaders

All loaders live in `langchain_community.document_loaders` (install: `pip install langchain-community`). Every loader outputs `Document(page_content: str, metadata: dict)` objects.

### Core pattern

```python
# Eager: loads everything into memory
docs: list[Document] = loader.load()

# Lazy: memory-efficient generator
for doc in loader.lazy_load():
    process(doc)

# Async lazy:
async for doc in loader.alazy_load():
    await process(doc)
```

### PDF — `PyPDFLoader`

```python
from langchain_community.document_loaders import PyPDFLoader

# mode="page" (default): one Document per page
loader = PyPDFLoader("paper.pdf", mode="page")
docs = loader.load()
# docs[0].metadata → {"source": "paper.pdf", "page": 0, "total_pages": 42}

# mode="single": entire PDF as one Document
loader = PyPDFLoader("paper.pdf", mode="single")

# Extract images as well (requires pypdf[full]):
loader = PyPDFLoader("paper.pdf", extract_images=True)
```

Other PDF loaders: `PDFMinerLoader` (better text layout), `PDFPlumberLoader` (tables), `AmazonTextractPDFLoader` (OCR via AWS).

### Web — `WebBaseLoader`

```python
from langchain_community.document_loaders import WebBaseLoader
import bs4

# Single URL:
loader = WebBaseLoader("https://python.langchain.com/docs/introduction/")
docs = loader.load()

# Multiple URLs (fetched concurrently):
loader = WebBaseLoader(
    ["https://url1.com", "https://url2.com"],
    bs_kwargs={"parse_only": bs4.SoupStrainer(class_=("post-content", "article"))},
    # bs_kwargs: BeautifulSoup kwargs for HTML parsing
    # requests_kwargs: {"headers": {...}, "timeout": 30}
    # continue_on_failure=True: skip URLs that fail
)
```

For JavaScript-rendered content, use `AsyncChromiumLoader` (requires `playwright`).

### Directory — `DirectoryLoader`

```python
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader

loader = DirectoryLoader(
    path="data/",
    glob="**/*.pdf",           # glob pattern
    loader_cls=PyPDFLoader,
    loader_kwargs={"mode": "page"},
    recursive=True,            # recurse subdirectories
    show_progress=True,        # tqdm progress bar
    use_multithreading=True,   # parallel loading
    max_concurrency=4,
    silent_errors=True,        # skip files that fail to load
)
docs = loader.load()
```

Auto-detect loader by file extension:
```python
from langchain_community.document_loaders import DirectoryLoader

loader = DirectoryLoader("data/")   # uses UnstructuredFileLoader by default for unknown types
```

### Other common loaders

```python
from langchain_community.document_loaders import (
    TextLoader,          # plain .txt files
    CSVLoader,           # CSV → one Document per row
    UnstructuredMarkdownLoader,  # .md files with structure
    UnstructuredHTMLLoader,      # HTML files
    JSONLoader,          # JSON with jq path extraction
    NotionDBLoader,      # Notion database
    ConfluenceLoader,    # Confluence pages
    GitLoader,           # Git repo files
    SlackDirectoryLoader,# Slack export
    YoutubeLoader,       # YouTube transcript
    AzureBlobStorageContainerLoader,
    S3DirectoryLoader,
    GoogleDriveLoader,
)

# CSV example — one doc per row:
from langchain_community.document_loaders.csv_loader import CSVLoader
loader = CSVLoader(
    file_path="data.csv",
    csv_args={"delimiter": ",", "fieldnames": ["id", "text", "label"]},
    source_column="id",      # used as metadata["source"]
)

# JSON example — extract text from nested JSON:
from langchain_community.document_loaders import JSONLoader
loader = JSONLoader(
    file_path="data.json",
    jq_schema=".items[].body",  # jq expression for text extraction
    metadata_func=lambda rec, meta: {**meta, "id": rec.get("id")},
)
```

---

## 2. Text Splitters

Install: `pip install langchain-text-splitters`

All splitters take a list of `Document` or `str` objects and output `list[Document]`. The key parameters are `chunk_size` (max chars/tokens per chunk) and `chunk_overlap` (overlap between consecutive chunks to preserve context).

### `RecursiveCharacterTextSplitter` (recommended default)

Tries separators in order: `["\n\n", "\n", " ", ""]`. Splits on the largest natural boundary first, falling back to smaller ones.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,       # target chars per chunk
    chunk_overlap=150,     # overlap chars between chunks
    length_function=len,   # how to measure chunk size
    is_separator_regex=False,
    separators=["\n\n", "\n", " ", ""],  # default; customize for your domain
    add_start_index=True,  # add metadata["start_index"] to each chunk
)

chunks: list[Document] = splitter.split_documents(docs)
# or for strings:
chunks = splitter.create_documents(["text1...", "text2..."],
                                    metadatas=[{"src": "a"}, {"src": "b"}])
```

### `TokenTextSplitter` — token-aware splitting (recommended for fitting context windows)

```python
from langchain_text_splitters import TokenTextSplitter

splitter = TokenTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base",    # OpenAI's tokenizer (good proxy for most models)
    chunk_size=512,                  # tokens per chunk
    chunk_overlap=64,
)
# For Gemini, there's no official tokenizer splitter yet —
# cl100k_base is a reasonable approximation.
```

### Language-aware splitters

```python
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

# Use language-specific separators (functions, classes, etc.)
py_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=500,
    chunk_overlap=50,
)
md_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.MARKDOWN,
    chunk_size=1000, chunk_overlap=100,
)
```

Other `Language` values: `JS`, `TS`, `JAVA`, `GO`, `RUBY`, `RUST`, `CPP`, `HTML`, `LATEX`, `SOL`.

### `MarkdownHeaderTextSplitter` — split by heading hierarchy

```python
from langchain_text_splitters import MarkdownHeaderTextSplitter

headers_to_split_on = [
    ("#",  "Header 1"),
    ("##", "Header 2"),
    ("###","Header 3"),
]
splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on,
    strip_headers=False,   # keep header text in chunk content
    return_each_line=False,
)
splits = splitter.split_text(markdown_string)
# metadata["Header 1"] = "Chapter Title", metadata["Header 2"] = "Section Title", etc.
```

### `SemanticChunker` — content-aware splitting (experimental)

Splits where embedding distance between consecutive sentences is high (semantic boundary):

```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_google_genai import GoogleGenerativeAIEmbeddings

chunker = SemanticChunker(
    GoogleGenerativeAIEmbeddings(model="gemini-embedding-001",
                                  task_type="SEMANTIC_SIMILARITY"),
    breakpoint_threshold_type="gradient",    # "percentile" | "standard_deviation" | "gradient"
    breakpoint_threshold_amount=95,          # percentile for "percentile" type
)
chunks = chunker.split_documents(docs)
```

Slower than character splitters (makes embedding API calls) but produces more semantically coherent chunks.

### Chunking strategy guidance

| Content type | Recommended splitter | `chunk_size` guideline |
|---|---|---|
| General text / prose | `RecursiveCharacterTextSplitter` | 800–1200 chars |
| Code | `RecursiveCharacterTextSplitter.from_language(Language.X)` | 500–800 chars |
| Markdown / docs | `MarkdownHeaderTextSplitter` then recursive | sections + 1000 chars |
| Structured JSON | `RecursiveJsonSplitter` | based on object size |
| Token-budget critical | `TokenTextSplitter` | 256–512 tokens |
| Domain with clear semantic boundaries | `SemanticChunker` | N/A (auto) |

Rule of thumb: the embedding model's ideal input length (for `gemini-embedding-001`) is 512–1024 tokens. Very short chunks lose context; very long chunks dilute the embedding signal. Start with `chunk_size=1000, chunk_overlap=150`.

---

## 3. Embeddings

See **`01-gemini-models-and-embeddings.md`** for full `GoogleGenerativeAIEmbeddings` reference.

Key rule: use **different `task_type` values** for indexing vs. querying:
- Indexing: `task_type="RETRIEVAL_DOCUMENT"`
- Querying: `task_type="RETRIEVAL_QUERY"`

```python
from langchain_google_genai import GoogleGenerativeAIEmbeddings

doc_emb = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001", task_type="RETRIEVAL_DOCUMENT"
)
qry_emb = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001", task_type="RETRIEVAL_QUERY"
)
```

---

## 4. Vector Stores

All vector stores implement the same core interface:

```python
# Add documents (embeds + stores):
ids = vector_store.add_documents(chunks: list[Document]) -> list[str]
await vector_store.aadd_documents(chunks)

# Search by text (embeds query internally):
results: list[Document] = vector_store.similarity_search(query, k=4, filter={...})
results_with_scores: list[tuple[Document, float]] = vector_store.similarity_search_with_score(query, k=4)

# Search by pre-computed vector:
results = vector_store.similarity_search_by_vector(embedding_vector, k=4)

# MMR search (maximal marginal relevance — balances similarity and diversity):
results = vector_store.max_marginal_relevance_search(query, k=4, fetch_k=20, lambda_mult=0.5)

# Delete:
vector_store.delete(ids=["id1", "id2"])

# Get retriever:
retriever = vector_store.as_retriever(
    search_type="similarity",           # "similarity" | "mmr" | "similarity_score_threshold"
    search_kwargs={"k": 5},
    # or: search_kwargs={"k": 5, "score_threshold": 0.7}  (for threshold type)
    # or: search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.5}  (for mmr)
)
```

### FAISS (local, no persistence by default)

```python
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

dim = 3072   # or output_dimensionality setting from embeddings
vs = FAISS(
    embedding_function=doc_emb,
    index=faiss.IndexFlatL2(dim),
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)
vs.add_documents(chunks)

# Save and load:
vs.save_local("faiss_index")
loaded = FAISS.load_local("faiss_index", embeddings=doc_emb,
                           allow_dangerous_deserialization=True)

# Build from documents directly:
vs = FAISS.from_documents(chunks, embedding=doc_emb)
```

FAISS is best for: local dev/testing, read-heavy workloads, simple L2/cosine similarity with no metadata filtering.

### Chroma (local with persistence, metadata filtering)

```python
from langchain_chroma import Chroma

# Persistent (saves to disk):
vs = Chroma(
    collection_name="my_docs",
    embedding_function=doc_emb,
    persist_directory="./chroma_db",
)
vs.add_documents(chunks)

# Metadata filtering:
results = vs.similarity_search(
    "LangGraph state",
    k=5,
    filter={"source": "tutorial.pdf"},          # exact match
    # filter={"year": {"$gte": 2024}},          # range filter
    # filter={"$and": [{"type": "doc"}, ...]},  # compound filter
)

# Build from documents:
vs = Chroma.from_documents(chunks, embedding=doc_emb,
                            persist_directory="./chroma_db")
```

Chroma is best for: local dev with metadata filtering, multi-collection setups.

### Pinecone (cloud, production)

```python
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

# Create index (once):
pc = Pinecone(api_key="...")
pc.create_index(
    name="rag-index",
    dimension=3072,        # must match embedding dimension
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
)

vs = PineconeVectorStore(
    embedding=doc_emb,
    index=pc.Index("rag-index"),
    namespace="default",
)
vs.add_documents(chunks)

# With metadata filtering:
results = vs.similarity_search("query", k=5,
                                filter={"doc_type": {"$eq": "policy"}})
```

### Other vector stores (same interface)

| Store | Install | Notes |
|---|---|---|
| `Weaviate` | `langchain-weaviate` | Graph-aware, rich filtering |
| `Qdrant` | `langchain-qdrant` | Fast, self-hosted or cloud |
| `PGVector` | `langchain-postgres` | PostgreSQL extension; good if you're already on Postgres |
| `Milvus` | `langchain-milvus` | High-scale self-hosted |
| `Redis` | `langchain-redis` | Low-latency, vector + keyword hybrid |
| `OpenSearch` | `langchain-opensearch` | AWS managed, hybrid BM25+vector |
| `MongoDB Atlas` | `langchain-mongodb` | Managed cloud |

---

## 5. Retrievers

Retrievers are `Runnable`s: `.invoke(query: str) -> list[Document]`.

### Basic vector store retriever

```python
retriever = vs.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 6, "fetch_k": 30, "lambda_mult": 0.7},
)
docs = retriever.invoke("What is LangGraph?")
```

### `MultiQueryRetriever` — rewrite query into N variants

Generates N alternative phrasings of the query, retrieves docs for each, and deduplicates by document ID:

```python
from langchain.retrievers.multi_query import MultiQueryRetriever

mq_retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=ChatGoogleGenerativeAI(model="gemini-2.5-flash"),
    include_original=True,     # also retrieve with the original query
    # num_queries=3,           # number of alternative queries to generate
)
docs = mq_retriever.invoke("How does LangGraph handle state?")
```

### `EnsembleRetriever` — combine multiple retrievers (BM25 + dense)

Uses Reciprocal Rank Fusion (RRF) to merge results from multiple retrievers. Classic combo: sparse (BM25/keyword) + dense (embedding):

```python
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

# BM25 needs all docs upfront (in-memory):
bm25 = BM25Retriever.from_documents(chunks)
bm25.k = 8

dense = vs.as_retriever(search_kwargs={"k": 8})

hybrid = EnsembleRetriever(
    retrievers=[bm25, dense],
    weights=[0.4, 0.6],     # sparse gets 40%, dense gets 60% weight
)
docs = hybrid.invoke("LangGraph state management")
```

### `ContextualCompressionRetriever` — compress/filter retrieved docs

Wraps a base retriever with a "compressor" that filters or shortens retrieved documents:

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import (
    LLMChainExtractor,
    EmbeddingsFilter,
    EmbeddingsRedundantFilter,
    DocumentCompressorPipeline,
)

# Extract only the relevant parts (uses LLM — adds latency):
llm_extractor = LLMChainExtractor.from_llm(
    ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")  # use cheap model
)
compressed = ContextualCompressionRetriever(
    base_compressor=llm_extractor,
    base_retriever=hybrid,
)

# Filter by embedding similarity to query (no LLM call):
emb_filter = EmbeddingsFilter(
    embeddings=qry_emb,
    similarity_threshold=0.76,
)
compressed_emb = ContextualCompressionRetriever(
    base_compressor=emb_filter,
    base_retriever=hybrid,
)

# Pipeline: first deduplicate, then filter by similarity:
pipeline = DocumentCompressorPipeline(transformers=[
    EmbeddingsRedundantFilter(embeddings=doc_emb),
    EmbeddingsFilter(embeddings=qry_emb, similarity_threshold=0.76),
])
compressed_pipeline = ContextualCompressionRetriever(
    base_compressor=pipeline,
    base_retriever=hybrid,
)
```

### `ParentDocumentRetriever` — small chunks for search, large chunks for context

Stores small chunks in the vector DB (for precise matching) but returns the larger parent chunks to the LLM (for context):

```python
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryByteStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
child_splitter  = RecursiveCharacterTextSplitter(chunk_size=400,  chunk_overlap=50)

# The docstore holds the full parent documents:
docstore = InMemoryByteStore()   # for production: use RedisStore, LocalFileStore, etc.

retriever = ParentDocumentRetriever(
    vectorstore=vs,
    docstore=docstore,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,   # if None, full docs are stored as parents
    id_key="doc_id",
)
retriever.add_documents(full_docs)    # NOT add_documents on the vector store
docs = retriever.invoke("What is LangGraph?")  # returns parent-sized chunks
```

### `MultiVectorRetriever` — multiple embeddings per document

Embed different representations of the same document (summary, hypothetical questions, full text) to improve recall:

```python
from langchain.retrievers import MultiVectorRetriever
from langchain.storage import InMemoryByteStore
import uuid, json

retriever = MultiVectorRetriever(
    vectorstore=vs,
    docstore=InMemoryByteStore(),
    id_key="doc_id",
)

# Generate summaries with LLM, store original + summary embeddings:
for doc in full_docs:
    doc_id = str(uuid.uuid4())
    doc.metadata["doc_id"] = doc_id

    summary = llm.invoke(f"Summarize: {doc.page_content}").content
    summary_doc = Document(page_content=summary, metadata={"doc_id": doc_id})

    retriever.vectorstore.add_documents([summary_doc])
    retriever.docstore.mset([(doc_id, doc)])
```

### `SelfQueryRetriever` — LLM extracts metadata filters

Uses LLM to extract structured metadata filters from a natural-language query:

```python
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.schema import AttributeInfo

metadata_field_info = [
    AttributeInfo(name="source", description="The PDF file name", type="string"),
    AttributeInfo(name="page", description="Page number", type="integer"),
    AttributeInfo(name="year", description="Publication year", type="integer"),
]

sq_retriever = SelfQueryRetriever.from_llm(
    llm=ChatGoogleGenerativeAI(model="gemini-2.5-flash"),
    vectorstore=vs,
    document_content_description="Research papers on AI agents",
    metadata_field_info=metadata_field_info,
    verbose=True,
)
docs = sq_retriever.invoke("Papers about RAG published after 2024 from attention.pdf")
# LLM extracts: query="RAG", filter={"year": {"$gt": 2024}, "source": "attention.pdf"}
```

---

## 6. Reranking

After retrieving a large initial set, reranking orders them by relevance. The two main approaches:

### LLM-based reranking (using `ContextualCompressionRetriever`)

```python
from langchain.retrievers.document_compressors import LLMChainExtractor

reranker_compressor = LLMChainExtractor.from_llm(
    ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite"),  # cheap model
)
reranked = ContextualCompressionRetriever(
    base_compressor=reranker_compressor,
    base_retriever=vs.as_retriever(search_kwargs={"k": 20}),  # fetch many, rerank to top
)
```

### Cohere Reranker

```python
from langchain_cohere import CohereRerank

reranker = CohereRerank(model="rerank-v3.5", top_n=5)
compressed = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=vs.as_retriever(search_kwargs={"k": 20}),
)
```

### Flashrank (local, free)

```python
from langchain_community.document_compressors import FlashrankRerank

reranker = FlashrankRerank(model="ms-marco-MiniLM-L-12-v2", top_n=5)
```

---

## 7. Complete RAG Chain (LCEL)

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── 1. Ingest ──────────────────────────────────────────────────────────
docs = PyPDFLoader("handbook.pdf").load()
chunks = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=150
).split_documents(docs)

# ── 2. Embed + index ───────────────────────────────────────────────────
doc_emb = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001", task_type="RETRIEVAL_DOCUMENT"
)
vs = Chroma.from_documents(chunks, embedding=doc_emb, persist_directory="./chroma")

# ── 3. Retriever ───────────────────────────────────────────────────────
retriever = vs.as_retriever(search_type="mmr", search_kwargs={"k": 6, "fetch_k": 20})

# ── 4. Prompt ──────────────────────────────────────────────────────────
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an expert assistant. Answer ONLY from the provided context. "
     "If the context doesn't contain the answer, say so.\n\n"
     "Context:\n{context}"),
    ("human", "{question}"),
])

# ── 5. LLM ────────────────────────────────────────────────────────────
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0)

# ── 6. Chain ───────────────────────────────────────────────────────────
def format_docs(docs):
    return "\n\n---\n\n".join(
        f"[Source: {d.metadata.get('source','?')}, Page: {d.metadata.get('page','?')}]\n"
        f"{d.page_content}"
        for d in docs
    )

rag_chain = (
    RunnableParallel(
        context=retriever | format_docs,
        question=RunnablePassthrough(),
    )
    | prompt
    | llm
    | StrOutputParser()
)

answer = rag_chain.invoke("What is the vacation policy?")

# ── 7. Stream ──────────────────────────────────────────────────────────
for chunk in rag_chain.stream("What is the vacation policy?"):
    print(chunk, end="", flush=True)

# ── 8. With sources (return docs alongside answer) ─────────────────────
rag_chain_with_sources = (
    RunnableParallel(
        context=retriever | format_docs,
        question=RunnablePassthrough(),
        source_docs=retriever,          # keep original docs too
    )
    | RunnablePassthrough.assign(
        answer=({"context": lambda d: d["context"],
                 "question": lambda d: d["question"]}
                | prompt | llm | StrOutputParser())
    )
)
out = rag_chain_with_sources.invoke("What is the vacation policy?")
print(out["answer"])
for doc in out["source_docs"]:
    print(f"  - {doc.metadata['source']} p{doc.metadata.get('page','?')}")
```

---

## 8. Conversational RAG (as LangGraph)

For multi-turn RAG with history, build a `StateGraph` rather than a chain:

```python
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver

class RAGState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    retrieved_docs: list        # last retrieved documents

# ── Query rewriting node ───────────────────────────────────────────────
rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system", "Rewrite the user's question to be self-contained given the "
               "conversation history. Return ONLY the rewritten question."),
    MessagesPlaceholder("messages"),
])
rewriter = rewrite_prompt | llm | StrOutputParser()

def rewrite_query(state: RAGState) -> dict:
    if len(state["messages"]) <= 1:
        return {}   # no history to incorporate
    rewritten = rewriter.invoke({"messages": state["messages"]})
    return {}   # we'll use rewritten query in the retrieve step
                # (or pass it via state)

# ── Retrieve node ──────────────────────────────────────────────────────
def retrieve(state: RAGState) -> dict:
    last_human = next(
        (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None
    )
    query = last_human.content if last_human else ""
    docs = retriever.invoke(query)
    return {"retrieved_docs": docs}

# ── Generate node ──────────────────────────────────────────────────────
gen_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer using the context. Context:\n{context}"),
    MessagesPlaceholder("messages"),
])

def generate(state: RAGState) -> dict:
    context = format_docs(state["retrieved_docs"])
    chain = gen_prompt | llm
    ai_msg = chain.invoke({"context": context, "messages": state["messages"]})
    return {"messages": [ai_msg]}

# ── Graph ──────────────────────────────────────────────────────────────
with SqliteSaver.from_conn_string("rag.sqlite") as ckpt:
    graph = (
        StateGraph(RAGState)
        .add_node("retrieve", retrieve)
        .add_node("generate", generate)
        .add_edge(START, "retrieve")
        .add_edge("retrieve", "generate")
        .add_edge("generate", END)
        .compile(checkpointer=ckpt)
    )

    cfg = {"configurable": {"thread_id": "user-session-1"}}
    result = graph.invoke(
        {"messages": [HumanMessage(content="What is the vacation policy?")]}, cfg
    )
    # Follow-up — history is preserved:
    result = graph.invoke(
        {"messages": [HumanMessage(content="And sick leave?")]}, cfg
    )
```

---

## 9. Advanced Patterns

### Hypothetical Document Embeddings (HyDE)

Generate a hypothetical answer, embed it, then search with that embedding:

```python
from langchain_core.runnables import RunnableLambda

hyde_chain = (
    ChatPromptTemplate.from_template(
        "Write a short passage that would answer: {question}"
    )
    | llm
    | StrOutputParser()
    | RunnableLambda(lambda hyp: retriever.invoke(hyp))
)
docs = hyde_chain.invoke({"question": "What is HyDE?"})
```

### Step-back prompting

Generate a more abstract question, retrieve for that, then answer the original:

```python
stepback_chain = (
    RunnableParallel(
        original_docs=retriever,
        abstract_docs=(
            ChatPromptTemplate.from_template(
                "Generate a broader question: {question}"
            ) | llm | StrOutputParser() | retriever
        ),
    )
    | RunnableLambda(lambda d: d["original_docs"] + d["abstract_docs"])
)
```

### Adaptive/self-corrective RAG (LangGraph pattern)

Add a grading node that checks retrieved docs for relevance and loops back to retrieve with a rewritten query if they're poor:

```python
def grade_docs(state: RAGState) -> dict:
    """Grade retrieved docs. Return needs_rewrite=True if poor."""
    grader_llm = llm.with_structured_output({"type": "object", "properties": {
        "relevant": {"type": "boolean"}
    }})
    grades = [
        grader_llm.invoke(f"Is this doc relevant to the query? Doc: {d.page_content[:500]}")
        for d in state["retrieved_docs"]
    ]
    return {"needs_rewrite": not any(g["relevant"] for g in grades)}

def route_after_grade(state: RAGState) -> str:
    return "rewrite" if state.get("needs_rewrite") else "generate"

graph = (
    StateGraph(RAGState)
    .add_node("retrieve", retrieve)
    .add_node("grade", grade_docs)
    .add_node("rewrite", rewrite_query)
    .add_node("generate", generate)
    .add_edge(START, "retrieve")
    .add_edge("retrieve", "grade")
    .add_conditional_edges("grade", route_after_grade, {"rewrite": "rewrite", "generate": "generate"})
    .add_edge("rewrite", "retrieve")
    .add_edge("generate", END)
    .compile()
)
```

---

## 10. Gotchas

1. **`task_type` mismatch** is the #1 silent RAG quality killer. Always use `RETRIEVAL_DOCUMENT` for corpus embeddings and `RETRIEVAL_QUERY` for query embeddings.

2. **`chunk_overlap` too small** causes chunks to lose cross-boundary context. Use at least 10–15% of `chunk_size`.

3. **`k` too small** means relevant docs might not be in the top-k. Start with `k=6` and evaluate. For `ContextualCompressionRetriever`, fetch a larger initial set (`k=20`) and let the compressor filter it down.

4. **FAISS doesn't support metadata filtering.** If you need `filter={"category": "policy"}`, use Chroma, Pinecone, Qdrant, or PGVector.

5. **`ParentDocumentRetriever.add_documents`** must be called on the retriever (not the underlying vector store). The retriever handles splitting into child chunks and linking them to parent documents.

6. **`EnsembleRetriever` weights must sum to 1.0** if using RRF. If using score-based fusion, they're relative importance weights.

7. **Streaming through a retriever:** retrievers don't stream — they return a list. Streaming happens at the LLM step. Use `astream_events` with `include_types=["chat_model"]` to get token-by-token output from a RAG chain.

8. **Large PDFs:** for PDFs with >100 pages, use `lazy_load()` to avoid loading everything into memory at once.

9. **Deduplication:** `MultiQueryRetriever` and `EnsembleRetriever` both deduplicate by document `id` (from `metadata`). Ensure your documents have consistent `id` fields if you want reliable dedup.
