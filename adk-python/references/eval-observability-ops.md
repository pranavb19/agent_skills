# Evaluation, observability, safety, cost & 2.0 migration

## Evaluation (`adk eval`)

Use `.test.json` / EvalSet files (or build via the dev UI Eval tab). Two foundational metrics:
- **`tool_trajectory_avg_score`** — did the agent call the right tools in the right order?
  Supports EXACT / IN_ORDER / ANY_ORDER; **uses EXACT by default**, and **defaults to a 1.0
  threshold (100% match)**.
- **`response_match_score`** — ROUGE-1 word overlap vs a reference; **defaults to a 0.8
  threshold** (small margin allowed).

2.0-era criteria add:
- `final_response_match_v2` — LLM-as-judge for semantic equivalence; configurable via
  `judge_model_options`, e.g. `{"judge_model": "gemini-2.5-flash", "num_samples": 5}`.
- `hallucinations_v1`, `safety_v1`, and rubric-based custom metrics.

Integrate into CI via pytest. **Built-in eval scores the root agent's final response**, so
sub-agent quality needs separate span-level capture (see tracing).

## Observability & tracing

ADK is instrumented with **OpenTelemetry**. Export to:
- **Cloud Trace** — set `trace_to_cloud=True` (on `get_fast_api_app` or Agent Engine).
- Third parties: Arize AX, LangWatch, Datadog, FutureAGI, Maxim.

Traces give per-step trajectories, tool selection, handoff quality, and token/cost/latency per
agent. The `adk web` Trace tab shows the event graph, request/response per LLM call, tool
calls, and state transitions; the Events tab shows callbacks firing and `state_update` deltas.

**Production gap:** ADK's built-in eval targets the dev loop. For live-traffic quality drift,
per-agent cost attribution, and continuous scoring, add an external observability platform and
sample 5–10% of traffic for async eval.

## Safety / guardrails

- `before_model_callback` — input filtering / prompt-injection checks.
- `after_model_callback` — output moderation / PII redaction.
- `before_tool_callback` — authorization.
- Prefer **Plugins** (App/Runner level) for cross-agent policy (`configuration.md`).
- Consider managed safety (Model Armor). Search/Vertex-Search grounding attach citations.

## Error handling (2.0)

The framework now auto-catches exceptions to enable automatic retries, telemetry, and HITL
pauses. Therefore:
- **Do NOT wrap tool bodies in broad `except Exception:`** — it masks failures and disables
  2.0 auto-retry.
- **Never catch `BaseException`** — it traps `NodeInterruptedError` and breaks HITL pausing.

## Cost / token optimization

- Tune `RunConfig.get_session_config(num_recent_events=...)`; use context caching and
  compaction.
- Cap loops with `max_llm_calls` (does NOT protect `run_live` BIDI).
- Use cheaper models (Flash) for routing/simple steps, Pro for hard reasoning.
- Avoid per-turn instruction/string injection in callbacks (token bloat).
- ADK auto-manages context (filters irrelevant events, summarizes old turns, lazy-loads
  artifacts, tracks tokens).

## Concurrency

Runner/services are stateless and thread-safe; scale horizontally on a shared persistent
SessionService.

## ADK 2.0 migration gotchas

- **Yield events from nodes** — don't `enqueue_event` or append directly.
- **Update custom session DB schemas** for the new `node_info` / `output` event fields.
- **Update strict-JSON downstream validators** — `additionalProperties:false` will reject 2.0
  events.
- **Move custom execution logic out of `_run_async_impl` overrides** into callbacks or the new
  node model (the graph engine bypasses legacy overrides).
- **Python 3.11+** required for 2.0 features.
- **Session compatibility:** ADK 2.0 sessions are readable by ADK 1.28+ (extra fields ignored)
  but incompatible with older 1.x.

## Caveats (verify against your installed version)

- PyPI shows `google-adk 2.2.0` while docs badge features "v2.0.0"; confirm exact constructor
  signatures from the installed version's API reference.
- DB specifics are version-dependent: v1.22.0 session schema migration; Postgres+asyncpg
  timezone bug (#4366); pool-leak under error load (#3328).
- Issue-sourced behaviors may be patched: multimodal lost through AgentTool (#729); AdkApp
  multimodal dict requirement (#930); `output_schema` disables tools (#701); ValidationError
  on invalid structured output (#3759); MCP schema conversion failures (#1055).
- The "one built-in tool per agent" rule and its `bypass_multi_tools_limit` workaround (≥1.16)
  are version-dependent.
