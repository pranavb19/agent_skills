# Tools (deep dive)

## Table of contents
1. Function tools — mandatory syntax
2. ToolContext
3. Long-running tools
4. Built-in tools and the one-per-agent limit
5. Conflicting / ambiguous tools — disambiguation & best practices
6. OpenAPI / MCP / third-party / Google Cloud tools
7. Tool authentication

---

## 1. Function tools — mandatory syntax

Any Python function (sync or `async`) can be a tool. The LLM relies **entirely** on the
function **name, type hints, and docstring** to decide when and how to call it.

Rules:
- Use verb-noun names (`get_weather`), not `do_stuff`.
- All params must be JSON-serializable; **no default values** (the LLM supplies all args).
- Return a **dict**, ideally with a `status` key (`"success"`/`"error"`); non-dict returns
  get wrapped under `result`.
- Document every parameter and the return structure — but **never document `tool_context`**
  (it is injected after the LLM decides to call; mentioning it confuses the model).
- You **cannot** pass raw bytes through params — use the artifact pattern (`inputs.md`).

```python
def get_weather(city: str, units: str) -> dict:
    """Gets the current weather for a city.

    Use when the user asks about current conditions or temperature.

    Args:
        city: City name, e.g. "Paris".
        units: "metric" or "imperial".
    Returns:
        dict with status, temperature, and conditions.
    """
    ...
    return {"status": "success", "temperature": 21, "conditions": "clear"}
```

---

## 2. ToolContext

Add `tool_context: ToolContext` as the **last** parameter to gain:
- `state` — tracked read/write (auto-deltas).
- `actions` — `escalate`, `skip_summarization`, `transfer_to_agent`.
- `function_call_id`.
- `save_artifact` / `load_artifact` / `list_artifacts`.
- auth helpers — `request_credential`, `get_auth_response`.
- access to user content (multimodal).

```python
from google.adk.tools import ToolContext

def add_to_cart(item: str, quantity: int, tool_context: ToolContext) -> dict:
    """Add an item to the shopping cart.

    Args:
        item: product name. quantity: number to add.
    Returns: dict with status and the updated cart.
    """
    cart = tool_context.state.get("cart", {})
    cart[item] = cart.get(item, 0) + quantity
    tool_context.state["cart"] = cart      # tracked → persisted delta
    return {"status": "success", "cart": cart}
```

To break a `LoopAgent`:
```python
def exit_loop(tool_context: ToolContext) -> dict:
    """Signals that the task is complete and the loop should stop."""
    tool_context.actions.escalate = True
    return {"status": "done"}
```

---

## 3. Long-running tools

`LongRunningFunctionTool` wraps tools for human-in-the-loop / multi-hour operations; the tool
can return a pending status and the framework supports resuming.

---

## 4. Built-in tools and the one-per-agent limit

Built-ins: `google_search`, `BuiltInCodeExecutor` (code execution), `VertexAiSearchTool`,
URL context, RAG/retrieval.

**Hard limitations (must-know):**
- Only **one** built-in tool per agent; cannot be combined with custom function tools in the
  same agent.
- Built-in tools cannot be used in sub-agents (exception: `GoogleSearchTool` /
  `VertexAiSearchTool` via the ≥1.16 `bypass_multi_tools_limit=True` workaround).
- Built-in tools require **Gemini** models.

**Canonical multi-tool pattern** — isolate each built-in in its own agent, expose via AgentTool:

```python
from google.adk.tools import google_search
from google.adk.code_executors import BuiltInCodeExecutor
from google.adk.tools.agent_tool import AgentTool

search_agent = Agent(model="gemini-2.5-flash", name="SearchAgent",
                     description="Searches the web.", tools=[google_search])
coding_agent = Agent(model="gemini-2.5-flash", name="CodeAgent",
                     description="Runs Python.", code_executor=BuiltInCodeExecutor())

root = Agent(name="Root", model="gemini-2.5-flash",
             description="Coordinator with search and code abilities.",
             tools=[AgentTool(agent=search_agent), AgentTool(agent=coding_agent)])
```

---

## 5. Conflicting / ambiguous tools — disambiguation & best practices

The model selects tools purely from **names + descriptions + parameter schemas**.

- **Ambiguous or overlapping descriptions are the primary cause of wrong-tool selection.**
  If two tools have similar descriptions, the LLM may pick arbitrarily.
- **Name collisions:** two tools with the same function name (e.g. from two MCP servers, or a
  function and a wrapped agent) collide — ADK identifies tools by name, so duplicates lead to
  one shadowing the other or schema confusion.

Best practices to avoid conflicts:
- Make each tool's description state **what it does, when to use it, and its limitations**.
- Keep tools single-purpose; split big multi-purpose tools.
- Give disjoint, non-overlapping descriptions.
- In the agent instruction, explicitly map intents → tools ("for X use tool A; for Y use B").
- Prefer fewer, clearer tools over many similar ones.
- Use `tool_filter` on toolsets to expose only the needed subset; rename where possible.

---

## 6. OpenAPI / MCP / third-party / Google Cloud tools

- **OpenAPI tools** — `RestApiTool` / OpenAPI toolset auto-generate one tool per operation
  from a spec, with auth via OpenAPI security schemes.
- **MCP tools** (kept brief — see official MCP docs for details): `McpToolset` bridges to MCP
  servers (`StdioConnectionParams` for local child processes; `SseConnectionParams` /
  `StreamableHTTPConnectionParams` for remote). Use `tool_filter` to expose a subset. Define
  `root_agent` **synchronously** for deployment (avoid async `from_server` patterns). Some MCP
  servers produce schemas that fail ADK/Pydantic conversion (GitHub #1055).
- **Third-party** — `LangchainTool(...)` and `CrewaiTool(...)` wrap LangChain / CrewAI tools.
- **Google Cloud tools** — prebuilt integrations for BigQuery, AlloyDB, Spanner, Cloud SQL,
  Firestore, Bigtable; enterprise connectors (Salesforce, Workday, SAP); MCP Toolbox for
  Databases.

---

## 7. Tool authentication

Tools requiring auth use:
```python
cred = tool_context.get_auth_response(auth_config)
if not cred:
    tool_context.request_credential(auth_config)
    return {"status": "auth_required"}
# use cred; cache it in state for reuse within the session
```
Heavily used with OpenAPI security schemes. Store the retrieved credential in state for reuse
during the session.
