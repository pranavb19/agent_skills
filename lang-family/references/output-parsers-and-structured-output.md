# 03 — Output Parsers and Structured Output

Two ways to get structured data from a model:

1. **`model.with_structured_output(Schema)`** — preferred for all models that support tool-calling or native JSON schema. The model is constrained to produce valid structured output. More reliable, no parsing errors.
2. **Output parsers** — appended to a chain; parse free-text model output into objects. Fragile on models without native structure support. Use when `with_structured_output` is unavailable.

---

## `with_structured_output` (strongly preferred path for Gemini)

```python
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from typing import Optional, Literal

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

class ResearchNote(BaseModel):
    """A research note with key findings."""
    topic: str = Field(description="The main topic")
    summary: str = Field(description="2-3 sentence summary")
    confidence: Literal["low", "medium", "high"] = Field(description="Confidence level")
    sources: list[str] = Field(description="URLs or references used", default_factory=list)
    follow_up_questions: Optional[list[str]] = None

# Default: method="json_schema" (Gemini native — most reliable)
structured = llm.with_structured_output(ResearchNote)
note: ResearchNote = structured.invoke("Summarize the transformer architecture.")

# Explicit method parameter:
structured_fc = llm.with_structured_output(ResearchNote, method="function_calling")
```

### `method` parameter

| Value | Mechanism | When to use |
|---|---|---|
| `"json_schema"` | Gemini `response_json_schema` API — model is constrained to valid JSON matching the schema. | **Default for Gemini. Always use this.** |
| `"function_calling"` | Invokes a hidden tool whose args match the schema. | Fallback. Use when JSON schema has features the Gemini API rejects. |

### `include_raw=True` — get both raw message and parsed result

```python
structured_raw = llm.with_structured_output(ResearchNote, include_raw=True)
out = structured_raw.invoke("Summarize transformers.")
# out is a dict:
# {
#   "raw": AIMessage(content='', additional_kwargs={"tool_calls": [...]}, ...),
#   "parsed": ResearchNote(topic="Transformers", ...),
#   "parsing_error": None,     # or an exception if parsing failed
# }
```

Use `include_raw=True` in production to handle parsing failures gracefully instead of raising exceptions.

### Accepting TypedDict or dict (JSON schema)

```python
from typing_extensions import TypedDict

class MovieTyped(TypedDict):
    title: str
    year: int

# TypedDict form — less validation than Pydantic
structured = llm.with_structured_output(MovieTyped)

# Raw JSON schema dict — maximum flexibility
schema = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "year":  {"type": "integer"},
    },
    "required": ["title", "year"],
}
structured = llm.with_structured_output(schema)
result: dict = structured.invoke("Tell me about Inception.")
```

### Pydantic schema gotchas for Gemini `json_schema` mode

- `Union[X, None]` / `Optional[X]` → **works** (maps to nullable).
- Nested Pydantic models → **works**, `$defs` are auto-inlined.
- `Literal["a", "b", "c"]` → **works** (maps to enum).
- `list[str]`, `dict[str, Any]` → **works**.
- Deep nesting (>5 levels) → may fail with schema complexity errors; flatten if possible.
- Recursive models → **not supported**.
- `model_validator` / `field_validator` → parsed after structure, validators run as normal.

---

## `StrOutputParser`

Extracts the `.content` string from `AIMessage` or `AIMessageChunk`. The most common chain terminator.

```python
from langchain_core.output_parsers import StrOutputParser

chain = prompt | llm | StrOutputParser()
result: str = chain.invoke({"question": "What is Python?"})
```

Streaming: when used with `.stream()`, emits string fragments as they arrive.

---

## `JsonOutputParser`

Expects the model to output a JSON string (possibly wrapped in a markdown code block). Parses it into a Python `dict` / `list`. Supports streaming (yields partial dicts as JSON is built).

```python
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

parser = JsonOutputParser()

# Inject format instructions into the prompt:
prompt = PromptTemplate(
    template="Output a JSON object with keys 'name' and 'age'.\n{format_instructions}\nPerson: {input}",
    input_variables=["input"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

chain = prompt | llm | parser
result: dict = chain.invoke({"input": "Alice, 30 years old"})
# {"name": "Alice", "age": 30}
```

**Streaming with JsonOutputParser:**
```python
for chunk in chain.stream({"input": "Bob, 25"}):
    print(chunk)  # partial dicts emitted as tokens arrive
# {} → {"name": ""} → {"name": "Bob"} → {"name": "Bob", "age": 25}
```

**Limitation:** relies on the model actually producing valid JSON. Use `with_structured_output` instead for Gemini — it's constrained, not prompt-injected.

---

## `PydanticOutputParser`

Validates free-text JSON against a Pydantic model. Includes `.get_format_instructions()` to inject schema description into the prompt.

```python
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

class Joke(BaseModel):
    setup: str = Field(description="The question setting up the joke")
    punchline: str = Field(description="The punchline resolving the joke")

parser = PydanticOutputParser(pydantic_object=Joke)

prompt = PromptTemplate(
    template="Answer the user's request.\n{format_instructions}\nRequest: {query}",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

chain = prompt | llm | parser
joke: Joke = chain.invoke({"query": "Tell me a joke about cats"})
```

**When to use `PydanticOutputParser` vs `with_structured_output`:**

| Scenario | Use |
|---|---|
| Gemini (any variant) | `with_structured_output(Schema, method="json_schema")` |
| Any model with tool-calling support | `with_structured_output(Schema)` |
| Model without tool-calling (e.g., local GGUF via Ollama) | `PydanticOutputParser` |
| Need to parse pre-existing string output (not from a fresh LLM call) | `PydanticOutputParser` or `JsonOutputParser` |

---

## `XMLOutputParser`

For models that produce XML or when you want structured output as XML tags:

```python
from langchain_core.output_parsers import XMLOutputParser

parser = XMLOutputParser(tags=["movie", "title", "year", "director"])
prompt = PromptTemplate.from_template(
    "Return info about this movie as XML.\n{format_instructions}\nMovie: {title}"
)
prompt = prompt.partial(format_instructions=parser.get_format_instructions())
chain = prompt | llm | parser
result: dict = chain.invoke({"title": "Inception"})
```

---

## `CommaSeparatedListOutputParser`

```python
from langchain_core.output_parsers import CommaSeparatedListOutputParser

parser = CommaSeparatedListOutputParser()
chain = PromptTemplate.from_template(
    "List 5 {item}.\n{format_instructions}"
).partial(format_instructions=parser.get_format_instructions()) | llm | parser
result: list[str] = chain.invoke({"item": "programming languages"})
```

---

## `OutputFixingParser` — retry on parse failure

Wraps another parser; if parsing fails, calls the LLM again to fix the output:

```python
from langchain.output_parsers import OutputFixingParser

base_parser = PydanticOutputParser(pydantic_object=Joke)
fixing_parser = OutputFixingParser.from_llm(parser=base_parser, llm=llm)
```

---

## Chaining multiple parsers (rare)

```python
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableLambda

# Parse JSON then extract a field
chain = prompt | llm | JsonOutputParser() | RunnableLambda(lambda d: d["answer"])
```

---

## Using output parsers in LangGraph

In a LangGraph node, don't add a parser to the node function itself — instead use `with_structured_output` on the model before passing it into the graph:

```python
from langgraph.graph import MessagesState

structured_llm = llm.with_structured_output(MySchema)

def extraction_node(state: MessagesState):
    result: MySchema = structured_llm.invoke(state["messages"])
    return {"extracted_data": result.model_dump()}
```

Or call the parser explicitly inside the node:

```python
import json
from langchain_core.output_parsers import JsonOutputParser

parser = JsonOutputParser()

def parse_node(state: MessagesState):
    raw: str = llm.invoke(state["messages"]).content
    parsed = parser.parse(raw)   # parse() method works without being in a chain
    return {"parsed": parsed}
```

---

## Format instructions pattern

Every parser (except `StrOutputParser`) provides `.get_format_instructions() -> str`. This returns text you inject into the prompt to tell the model what format to produce. For Gemini with `with_structured_output`, you don't need format instructions — the model is constrained natively and the prompt stays clean.

```python
# Format instructions example (for non-Gemini or PydanticOutputParser path)
instructions = parser.get_format_instructions()
# "The output should be formatted as a JSON instance that conforms to the JSON schema below..."
```

---

## Summary decision tree

```
Does the model support tool-calling or native JSON schema?
 ├─ YES (Gemini, GPT-4, Claude) → model.with_structured_output(PydanticSchema)
 │    └─ For Gemini specifically → method="json_schema" (default, most reliable)
 │
 └─ NO (local models, older GPT) → PromptTemplate + PydanticOutputParser
      ├─ If output is consistently valid JSON → JsonOutputParser (simpler)
      └─ If output is flaky → wrap with OutputFixingParser
```
