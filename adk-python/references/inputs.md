# Inputs to agents & multi-agent systems (deep dive)

The universal input object is `types.Content(role="user", parts=[...])`, where each `Part`
is text, inline bytes, or a file URI. This is how you feed **text, multiple files, images,
audio, PDFs, and structured JSON**.

## 1. Text + multiple files / multimodal in one turn

```python
from google.genai import types

def build_message(text, image_path=None, pdf_path=None, audio_path=None):
    parts = [types.Part(text=text)]
    if image_path:
        parts.append(types.Part.from_bytes(
            data=open(image_path, "rb").read(), mime_type="image/jpeg"))
    if pdf_path:
        parts.append(types.Part.from_bytes(
            data=open(pdf_path, "rb").read(), mime_type="application/pdf"))  # Gemini reads PDFs natively
    if audio_path:
        parts.append(types.Part.from_bytes(
            data=open(audio_path, "rb").read(), mime_type="audio/mp3"))
    return types.Content(role="user", parts=parts)   # MULTIPLE files in one message

msg = build_message("Compare the chart in the image with the table in the PDF.",
                    image_path="chart.png", pdf_path="report.pdf")
# runner.run_async(user_id=..., session_id=..., new_message=msg)
```

Key facts:
- A single `Content` can hold **many** parts → that's how you pass multiple files at once.
  Order them logically (text framing first usually helps).
- **GCS files:** use `types.Part.from_uri(file_uri="gs://bucket/img.jpg", mime_type="image/jpeg")`
  instead of bytes (better for large files / Agent Engine).
- **Always set an accurate `mime_type`.** Wrong/missing MIME is a top cause of "the model
  ignored my file".
- **Use Gemini** for image/PDF/audio understanding — most non-Gemini LiteLLM models won't
  accept inline binary parts.
- **AdkApp / `async_stream_query` quirk (GitHub #930):** when querying a *deployed* agent, the
  multimodal `message` may need to be a **dict** rather than `types.Part` objects:
  ```python
  message = {"role": "user", "parts": [
      {"text": "Summarize the uploaded file."},
      {"inline_data": {"data": file_bytes, "mime_type": mime_type}},
  ]}
  ```

## 2. Binary data and TOOLS — the artifact pattern (must-know)

**You cannot pass raw bytes through tool parameters** (tools accept only simple JSON types).
The canonical solution: save uploads as **artifacts** and pass the artifact **filename**
(a string) to tools.

```python
from google.adk.tools import ToolContext
from google.adk.agents.callback_context import CallbackContext
from google.genai import types

# 1) before_model_callback: detect user-uploaded inline data, persist as artifact,
#    inject a text reference so the LLM can name it to tools.
def stash_uploads(callback_context: CallbackContext, llm_request):
    for content in llm_request.contents:
        for part in (content.parts or []):
            if part.inline_data and part.inline_data.mime_type.startswith("image/"):
                fname = f"usr_upl_img_{callback_context.invocation_id}.png"
                callback_context.save_artifact(fname, types.Part(inline_data=part.inline_data))
                content.parts.append(types.Part(text=f"[uploaded image saved as artifact: {fname}]"))
    return None  # proceed to the model

# 2) The tool takes the FILENAME (string), loads bytes from the artifact service.
def describe_image(artifact_filename: str, tool_context: ToolContext) -> dict:
    """Analyzes a previously uploaded image.

    Args:
        artifact_filename: The artifact name shown in context, e.g. "usr_upl_img_123.png".
    Returns:
        dict with status and a short description.
    """
    part = tool_context.load_artifact(artifact_filename)   # bytes back from ArtifactService
    # ... pass part to a vision model / processing ...
    return {"status": "success", "description": "a coffee cup on a desk"}
```

Why: keeps binary out of tool args, gives versioning, and lets the LLM choose *which*
uploaded file to act on by name.

## 3. Structured input (`input_schema`)

`input_schema` (a Pydantic `BaseModel`) declares the expected input contract — the agent then
expects a JSON string conforming to it.

```python
from pydantic import BaseModel, Field

class CountryInput(BaseModel):
    country: str = Field(description="Country to look up the capital of")

agent = Agent(model="gemini-2.5-flash", name="capital_agent",
              description="Returns a country's capital.",
              instruction="Given the country, return its capital.",
              input_schema=CountryInput, output_key="capital")
```

## 4. Passing inputs *between* agents (multi-agent)

- **Shared state (preferred):** agent A writes `output_key="parsed"`; agent B reads it via
  `{parsed}` in its instruction (works because workflow agents share the InvocationContext).
  Use `temp:` for intra-invocation-only data.
- **Files between agents:** save the file as an **artifact** in the first agent/tool, store
  the artifact name in state, load it downstream. Do NOT shove bytes through state.
- **AgentTool:** pass discrete structured input to a sub-expert and get a structured result
  back (caller keeps control).

See `outputs.md` for shaping what comes back out, and `memory-artifacts.md` for artifact
mechanics.
