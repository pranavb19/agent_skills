# Models & streaming

## Models

### Gemini (first-class)
Pass the model string directly (`"gemini-2.5-flash"`, `"gemini-flash-latest"`). Use an AI
Studio key, or Vertex via `GOOGLE_GENAI_USE_VERTEXAI=TRUE` + project/location env vars.
**Built-in tools and inline multimodal parts require Gemini.**

### LiteLLM (other providers)
```python
from google.adk.models.lite_llm import LiteLlm
agent  = Agent(model=LiteLlm(model="openai/gpt-4o"), name="gpt_agent", description="...")
agent2 = Agent(model=LiteLlm(model="anthropic/claude-3-5-sonnet-20241022"),
               name="claude_agent", description="...")
```
LiteLLM is an OpenAI-compatible bridge to 100+ models; set provider keys via env. Local models
via Ollama/vLLM (`LiteLlm(model="ollama_chat/llama3", api_base=...)` or an OpenAI-compatible
base URL). Note: most non-Gemini models cannot use `google_search`/code execution natively and
may reject inline binary multimodal parts.

### Other connectors
`ApigeeLlm` (Apigee AI Gateway), Vertex Model Garden.

### Model routing & fallback (2.0)
A router function selects models at runtime with automatic failover on error; you can also
implement try/except fallback manually (e.g., try Pro, fall back to Flash on error).

---

## Streaming

### SSE (server-sent events)
`RunConfig(streaming_mode=StreamingMode.SSE)` via `run_async`. For typewriter UIs: partial
events are marked `partial=True`, with a single final aggregated response; progressive
function-arg streaming and deferred parallel function execution are supported.

### Bidirectional live (BIDI) — real-time audio/video
`run_live` + `LiveRequestQueue` + `RunConfig(streaming_mode=StreamingMode.BIDI)` connects to
the Gemini Live API over WebSocket, with interruptions and voice-activity detection.

Key constraints:
- **One response modality per session** — `["TEXT"]` or `["AUDIO"]`, never both (API error).
  If unspecified, `run_live` defaults to `["AUDIO"]`.
- For multi-agent live, ADK auto-enables input+output audio transcription (needed for agent
  transfer context), even if set to None.
- Live API sessions are transient and time out after ~10 minutes of inactivity;
  `session_resumption=types.SessionResumptionConfig()` transparently reconnects so the
  conversation continues seamlessly; `context_window_compression` enables long sessions.
- `max_llm_calls` does NOT protect BIDI — add your own cost monitoring/guardrails.

### Streaming tools
Streaming tools can stream intermediate results (e.g., monitor a stock price / video stream
and react) rather than returning once.
