# Deployment

## Targets at a glance

| Target | Best for | Deploy |
|---|---|---|
| Vertex AI Agent Engine | Managed, least ops; inherits sessions/auth/trace/scaling | `adk deploy agent_engine ...` |
| Cloud Run | Serverless containers, custom routes | `adk deploy cloud_run ...` |
| GKE / custom containers | Full control, existing k8s | `adk deploy gke ...` |

## Vertex AI Agent Engine (managed)

```bash
adk deploy agent_engine --project=$PROJECT --region=$REGION \
  --display_name="My Agent" my_agent
```

Or programmatically:
```python
import vertexai
from vertexai import agent_engines
from vertexai.preview.reasoning_engines import AdkApp  # location may vary by version

app = AdkApp(agent=root_agent)
remote = agent_engines.create(agent_engine=app, requirements=["google-adk"])
# query:
async for ev in remote.async_stream_query(user_id="u_456", session_id=sid, message=msg):
    print(ev)
```
Inherits managed sessions, auth, Cloud Trace, and scaling. **Remember the multimodal-message
dict quirk** for `async_stream_query` (see `inputs.md`, GitHub #930).

## Cloud Run

```bash
adk deploy cloud_run --project=$PROJECT --region=$REGION my_agent
```
- Persist sessions via the `SESSION_SERVICE_URI` env (e.g., Cloud SQL Postgres — async driver!).
- Tune `--concurrency`.
- Endpoints: `/run`, `/run_sse`, `POST /apps/{app}/users/{u}/sessions/{s}`.
- For custom routes, build your own server with `get_fast_api_app` (kept brief — only when you
  need extra endpoints like `/feedback`).

## GKE / custom containers

`adk deploy gke` (auto image build + manifests) or manual `gcloud` + `kubectl`. Use
**Workload Identity** to grant `roles/aiplatform.user` (no JSON keys). Container runs
`uvicorn main:app`.

## Production sessions (critical)

- **Never use InMemory in production** — use Database / Vertex / Firestore so state survives
  restarts and is shared across instances/replicas (`sessions-state.md`).
- Consider session expiry/cleanup jobs.
- High-traffic apps may need distributed locking for concurrent same-session writes.

## Scaling

Runner and services are stateless and thread-safe; scale horizontally behind a shared
persistent SessionService. Use `get_session_config(num_recent_events=...)` and context
caching/compaction to bound per-request cost/latency (`eval-observability-ops.md`).
