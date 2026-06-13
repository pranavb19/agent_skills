# Outputs — getting the best, reliable output (deep dive)

The most reliable way to get parseable output is **`output_schema` + `output_key`**.

```python
import json
from pydantic import BaseModel, Field
from typing import List
from google.adk.agents import LlmAgent

class Product(BaseModel):
    name: str = Field(description="Product name")
    price: float = Field(description="Price in USD")

class ProductList(BaseModel):
    products: List[Product] = Field(description="All matching products")

extractor = LlmAgent(
    model="gemini-2.5-flash",
    name="extractor",
    description="Extracts a structured product list from free text.",
    instruction=(
        "Extract products from the user text. "
        "Return ONLY a JSON object matching the schema — no prose, no markdown fences."
    ),  # MANDATORY: also instruct the format; output_schema alone is not enough
    output_schema=ProductList,        # enforces controlled generation + validation
    output_key="product_result",      # stores the validated dict into session.state
    # NO tools= here — output_schema DISABLES tools and transfer
)
```

## Hard rules / gotchas

- Setting `output_schema` **disables tool use AND agent transfer** for that agent
  (GitHub #701). If you need tools *and* structure:
  - do the tool work in one agent, then pass results to a separate structuring agent
    (sequential pipeline), **or**
  - attach the schema to a tool's return value (tools and structure coexist that way —
    keep `output_schema` off the agent and have the tool return the structured dict).
- **On a sub-agent you MUST add** `disallow_transfer_to_parent=True,
  disallow_transfer_to_peers=True` — otherwise schema enforcement can be bypassed by a
  transfer attempt.
- **ADK strictly validates** the final response with `model_validate_json`. Invalid output
  raises `pydantic.ValidationError` — **wrap your run in try/except and provide a fallback**
  (GitHub #3759). Intermediate events (tool calls) are not subject to the final schema.
- **Always also instruct** the model to emit JSON only; rely on `Field(description=...)` to
  guide each field. `temperature=0.0` (via `generate_content_config`) helps determinism.
- The result lands in `state[output_key]` as a **dict**, ready for downstream agents/UI.

## try/except fallback pattern

```python
from pydantic import ValidationError

try:
    async for ev in runner.run_async(user_id=u, session_id=s, new_message=msg):
        if ev.is_final_response():
            result = ev.content.parts[0].text
except ValidationError:
    result = '{"products": []}'   # safe fallback / log + retry
```

## Nested / complex schemas

```python
class Attachment(BaseModel):
    filename: str
    content_type: str
    size_kb: int

class EmailWithAttachments(BaseModel):
    subject: str
    body: str
    priority: str
    attachments: List[Attachment]
# use as output_schema; output_key result is a fully nested dict in state
```

## When NOT to use output_schema

For chat-style answers where you still want occasional structure, keep `output_schema` off
and instead use a **tool that returns the structured dict** — tools and structure coexist
that way, and the agent keeps its ability to call tools and transfer.
