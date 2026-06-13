# Agent configuration & callbacks

## Configuration options

- **`model`** — a string (`"gemini-2.5-flash"`, `"gemini-flash-latest"`) or a wrapper object
  (`LiteLlm(...)`, `ApigeeLlm(...)`). See `models-streaming.md`.
- **`generate_content_config`** (`google.genai.types.GenerateContentConfig`): `temperature`,
  `top_p`, `max_output_tokens`, safety settings, etc. Use `temperature=0.0` for
  deterministic/structured extraction.
  ```python
  from google.genai import types
  agent = Agent(model="gemini-2.5-flash", name="x", description="...",
                generate_content_config=types.GenerateContentConfig(temperature=0.0,
                                                                     max_output_tokens=1024))
  ```
- **`instruction` vs `global_instruction`** — `instruction` is per-agent; `global_instruction`
  (set on the root) applies across the whole agent tree.
- **`input_schema` / `output_schema` / `output_key`** — see `inputs.md` and `outputs.md`.
  Remember: `output_schema` disables tools + transfer.
- **Planners** — `BuiltInPlanner(thinking_config=...)` leverages Gemini's native thinking;
  `PlanReActPlanner()` implements explicit Reason+Act for models without native thinking.
  Both subclass `BasePlanner`.
- **Code executors** — `BuiltInCodeExecutor` (Gemini native) or a custom `BaseCodeExecutor`.

## Callbacks — six lifecycle hooks

**Parameter names are fixed** (ADK passes by keyword): `callback_context` for agent/model
callbacks, `tool_context` for tool callbacks. Renaming raises `TypeError`.

| Callback | Fires | Return `None` | Return object → effect |
|---|---|---|---|
| `before_agent_callback` | before agent logic | proceed | `types.Content` → skip agent, use as output |
| `after_agent_callback` | after agent logic | keep output | `Content` → append |
| `before_model_callback` | before LLM call | proceed | `LlmResponse` → skip LLM (guardrail, cache) |
| `after_model_callback` | after LLM response | keep | `LlmResponse` → replace (sanitize, disclaimers) |
| `before_tool_callback` | before tool | proceed | `dict` → skip tool, use as result |
| `after_tool_callback` | after tool | keep | `dict` → replace tool result |

### Examples

Input guardrail (block + canned response):
```python
from google.adk.models import LlmResponse
from google.genai import types

def block_pii(callback_context, llm_request):
    text = llm_request.contents[-1].parts[0].text if llm_request.contents else ""
    if "ssn" in text.lower():
        return LlmResponse(content=types.Content(
            role="model", parts=[types.Part(text="I can't process sensitive personal data.")]))
    return None  # proceed
agent.before_model_callback = block_pii
```

Tool authorization:
```python
def authorize(tool, args, tool_context):
    if tool.name == "delete_account" and tool_context.state.get("user:role") != "admin":
        return {"status": "error", "reason": "not authorized"}  # skip the tool
    return None
agent.before_tool_callback = authorize
```

### Use cases
Input/output guardrails, caching, logging/observability, dynamic state, request/response
shaping, and stashing uploads as artifacts (`inputs.md`).

### Anti-patterns
- Don't do RAG retrieval in `before_model` — use a tool.
- Don't run long blocking work in `after_agent` — it's on the critical path.
- Don't append the same string to history every turn — token bloat.

### Plugins (preferred for cross-agent policy)
For security guardrails and policy at scale, prefer **Plugins** registered at the App/Runner
level over per-agent callbacks — more modular and applied consistently across the tree.
