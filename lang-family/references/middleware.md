# 07 — Agent Middleware

Middleware is the extension and customization mechanism for `create_agent` in LangChain v1. It provides lifecycle hooks that wrap the model calls, tool calls, and agent entry/exit — similar to HTTP middleware in web frameworks.

---

## Execution model

```
create_agent(model, tools, middleware=[mw1, mw2, mw3])

Lifecycle per agent run:
  before_agent (mw1 → mw2 → mw3)
    ↓
  [agent loop: model + tools]
    before_model (mw1 → mw2 → mw3)
    wrap_model_call (mw3 → mw2 → mw1 → ACTUAL MODEL CALL → mw1 → mw2 → mw3)
    after_model (mw1 → mw2 → mw3)
    wrap_tool_call (per tool call, same wrapping pattern)
    [loop back if more tool calls]
    ↓
  after_agent (mw3 → mw2 → mw1)   ← reversed on exit
```

Middleware is composable: each hook is called in order for the "enter" direction and reverse order for "exit", exactly like a stack.

---

## Import

```python
from langchain.agents.middleware import (
    # Built-in middleware:
    SummarizationMiddleware,
    HumanInTheLoopMiddleware,
    ModelCallLimitMiddleware,
    ToolCallLimitMiddleware,
    ModelFallbackMiddleware,
    PIIMiddleware,
    TodoListMiddleware,
    LLMToolSelectorMiddleware,
    ToolRetryMiddleware,
    ToolEmulator,
    ContextEditingMiddleware,
    ShellToolMiddleware,
    FilesystemSearchMiddleware,

    # Base class for custom middleware:
    AgentMiddleware,

    # Hook decorators for functional middleware:
    before_agent,
    before_model,
    wrap_model_call,
    after_model,
    wrap_tool_call,
    after_agent,
)
```

---

## Built-in middleware

### `SummarizationMiddleware`

Automatically summarizes the conversation history when it approaches the model's context limit. Prevents context overflow without requiring manual memory management.

```python
from langchain.agents.middleware import SummarizationMiddleware

mw = SummarizationMiddleware(
    model=None,                 # Uses the agent's model by default.
                                # Pass a different model (e.g., cheaper flash) for summaries.
    token_threshold=0.8,        # Summarize when messages exceed 80% of context limit.
    keep_last_n=10,             # Always preserve the last N messages verbatim.
    summary_prompt="Summarize the conversation so far, preserving key facts.",
)
```

### `HumanInTheLoopMiddleware`

Pauses the agent before executing tool calls and surfaces the pending call to a human for approval, editing, or rejection.

```python
from langchain.agents.middleware import HumanInTheLoopMiddleware

mw = HumanInTheLoopMiddleware(
    approval_required=True,     # If True, every tool call needs approval.
    tools_requiring_approval=["delete_file", "send_email"],  # OR: only specific tools.
    timeout_seconds=300,        # Seconds to wait for human response. None = no timeout.
    on_timeout="proceed",       # "proceed" | "abort" | "skip" when timeout fires.
)
```

When HITL triggers, the graph is interrupted and `state["__interrupt__"]` is set. Resume with:
```python
from langgraph.types import Command

# Approve (execute as planned):
result = agent.invoke(Command(resume={"action": "approve"}), config)

# Edit the tool call arguments:
result = agent.invoke(Command(resume={
    "action": "edit",
    "tool_call_id": "call_abc",
    "new_args": {"filename": "backup.txt"},  # override args
}), config)

# Reject (skip the tool call):
result = agent.invoke(Command(resume={"action": "reject"}), config)
```

### `ModelCallLimitMiddleware`

Hard cap on total model invocations within a single agent run:

```python
from langchain.agents.middleware import ModelCallLimitMiddleware

mw = ModelCallLimitMiddleware(
    limit=20,              # Max model calls per run. Raises AgentModelLimitError if exceeded.
    on_limit="raise",      # "raise" | "end" (gracefully end the run on limit)
)
```

### `ToolCallLimitMiddleware`

Hard cap on total tool invocations:

```python
from langchain.agents.middleware import ToolCallLimitMiddleware

mw = ToolCallLimitMiddleware(
    limit=50,
    per_tool_limits={"web_search": 10, "code_executor": 5},  # optional per-tool caps
    on_limit="end",
)
```

### `ModelFallbackMiddleware`

Automatically retries with fallback models on errors (rate limits, timeouts, context overflow):

```python
from langchain.agents.middleware import ModelFallbackMiddleware
from langchain_google_genai import ChatGoogleGenerativeAI

mw = ModelFallbackMiddleware(
    fallbacks=[
        ChatGoogleGenerativeAI(model="gemini-2.5-flash"),       # retry with cheaper model
        ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite"),  # last resort
    ],
    exceptions_to_catch=(TimeoutError, ResourceExhausted),  # which errors trigger fallback
    max_retries_per_model=2,
)
```

### `PIIMiddleware`

Detects and optionally redacts Personally Identifiable Information from inputs and outputs:

```python
from langchain.agents.middleware import PIIMiddleware

mw = PIIMiddleware(
    redact_inputs=True,       # Redact PII from messages before sending to model.
    redact_outputs=True,      # Redact PII from model outputs before returning.
    pii_types=["email", "phone", "ssn", "credit_card"],  # types to detect
    redaction_strategy="mask",  # "mask" (replace with ***) | "remove" | "tag"
    audit_log=True,            # Log detected PII to LangSmith metadata (not the values).
)
```

### `ToolRetryMiddleware`

Retries failed tool calls with LLM-assisted argument repair:

```python
from langchain.agents.middleware import ToolRetryMiddleware

mw = ToolRetryMiddleware(
    max_retries=3,
    retry_with_llm=True,    # Ask the LLM to fix the args before retrying.
    exception_types=(ValueError, TypeError),
)
```

### `LLMToolSelectorMiddleware`

Dynamically filters the available tools using an LLM based on the user's message. Reduces token usage for agents with many tools:

```python
from langchain.agents.middleware import LLMToolSelectorMiddleware

mw = LLMToolSelectorMiddleware(
    max_tools=5,       # Only pass the 5 most relevant tools per model call.
    selector_model=None,  # Use the agent's model. Pass a cheaper model to save cost.
)
```

### `TodoListMiddleware`

Maintains a persistent TODO list in agent state that survives across turns. Useful for long-running task agents:

```python
from langchain.agents.middleware import TodoListMiddleware

mw = TodoListMiddleware(
    inject_into_system_prompt=True,   # Automatically prepend TODO list to system prompt.
    state_key="todo_list",
)
```

### `ContextEditingMiddleware`

Allows the agent to edit its own context (system prompt, earlier messages) by emitting a special tool call:

```python
from langchain.agents.middleware import ContextEditingMiddleware

mw = ContextEditingMiddleware(
    allow_system_edit=True,
    allow_message_edit=True,
    max_edits_per_run=5,
)
```

### `ToolEmulator`

Emulates tools that are expensive or unavailable (useful in testing/dev):

```python
from langchain.agents.middleware import ToolEmulator

mw = ToolEmulator(
    emulated_tools={
        "send_email": lambda args: "Email sent successfully.",
        "delete_file": lambda args: f"[EMULATED] File {args['path']} deleted.",
    }
)
```

### `ShellToolMiddleware` / `FilesystemSearchMiddleware`

Built-in tools for shell command execution and filesystem search, exposed as middleware:

```python
from langchain.agents.middleware import ShellToolMiddleware, FilesystemSearchMiddleware

shell_mw = ShellToolMiddleware(
    allowed_commands=["ls", "cat", "head", "grep"],  # whitelist
    working_directory="/workspace",
    timeout_seconds=10,
)

search_mw = FilesystemSearchMiddleware(
    root_directory="/data",
    max_results=20,
)
```

---

## Combining middleware

Order matters: hooks fire in list order on the way in, reverse order on the way out.

```python
from langchain.agents import create_agent
from langchain.agents.middleware import (
    SummarizationMiddleware, HumanInTheLoopMiddleware,
    ModelCallLimitMiddleware, ToolCallLimitMiddleware,
    ModelFallbackMiddleware, PIIMiddleware,
)

agent = create_agent(
    model=ChatGoogleGenerativeAI(model="gemini-2.5-pro"),
    tools=tools,
    middleware=[
        PIIMiddleware(redact_inputs=True),          # 1st: clean inputs
        ModelFallbackMiddleware(fallbacks=[         # 2nd: handle model errors
            ChatGoogleGenerativeAI(model="gemini-2.5-flash")
        ]),
        SummarizationMiddleware(token_threshold=0.75),  # 3rd: manage context
        HumanInTheLoopMiddleware(                   # 4th: HITL for dangerous tools
            tools_requiring_approval=["delete", "email"]
        ),
        ModelCallLimitMiddleware(limit=30),         # 5th: safety caps
        ToolCallLimitMiddleware(limit=100),
    ],
    checkpointer=ckpt,          # Required for HITL (interrupt support)
    system_prompt="You are a careful assistant.",
)
```

---

## Custom middleware — decorator style

Use the hook decorators for simple functional middleware:

```python
from langchain.agents.middleware import before_model, after_model, wrap_model_call, wrap_tool_call
from langchain_core.messages import SystemMessage

@before_model
def inject_date_context(messages, config):
    """Add current date to system context before each model call."""
    from datetime import date
    date_msg = SystemMessage(content=f"Today is {date.today().isoformat()}.")
    return [date_msg] + messages  # return modified messages

@after_model
def log_tokens(ai_message, config):
    """Log token usage after each model call."""
    usage = ai_message.usage_metadata
    if usage:
        print(f"Tokens: prompt={usage['input_tokens']} completion={usage['output_tokens']}")
    return ai_message  # must return the message

@wrap_tool_call
def audit_tool_calls(tool_name, tool_args, call_next, config):
    """Audit log every tool invocation."""
    import logging
    logging.info(f"Tool: {tool_name}, Args: {tool_args}")
    result = call_next(tool_name, tool_args)  # execute the tool
    logging.info(f"Tool result: {result[:100]}")
    return result

@wrap_model_call
def cache_model_calls(messages, call_next, config):
    """Simple in-memory cache for model calls."""
    key = str(messages)
    if key in cache:
        return cache[key]
    result = call_next(messages)
    cache[key] = result
    return result

# Use as list of decorated functions (NOT instantiated — just the decorated fn):
agent = create_agent(
    model=llm, tools=tools,
    middleware=[inject_date_context, log_tokens, audit_tool_calls],
)
```

---

## Custom middleware — class style (`AgentMiddleware`)

For stateful or complex middleware:

```python
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.runnables import RunnableConfig

class TokenBudgetMiddleware(AgentMiddleware):
    """Track and enforce a per-session token budget."""

    def __init__(self, max_tokens_per_session: int = 100_000):
        self.max_tokens = max_tokens_per_session
        self._used: dict[str, int] = {}

    def before_agent(self, state: dict, config: RunnableConfig) -> dict:
        """Called once at the start of an agent run."""
        thread_id = config.get("configurable", {}).get("thread_id", "default")
        self._used.setdefault(thread_id, 0)
        return state  # must return state (can modify)

    def before_model(
        self,
        messages: list[BaseMessage],
        config: RunnableConfig,
    ) -> list[BaseMessage]:
        """Called before each model invocation."""
        thread_id = config.get("configurable", {}).get("thread_id", "default")
        if self._used.get(thread_id, 0) >= self.max_tokens:
            raise RuntimeError(f"Token budget exceeded for thread {thread_id}")
        return messages  # must return messages

    def after_model(
        self,
        ai_message: AIMessage,
        config: RunnableConfig,
    ) -> AIMessage:
        """Called after each model invocation."""
        thread_id = config.get("configurable", {}).get("thread_id", "default")
        usage = ai_message.usage_metadata or {}
        used = usage.get("total_tokens", 0)
        self._used[thread_id] = self._used.get(thread_id, 0) + used
        return ai_message  # must return the message

    def after_agent(self, state: dict, config: RunnableConfig) -> dict:
        """Called once at the end of an agent run (including on error)."""
        return state

    # Optional async hooks (used in async invocation):
    async def abefore_model(self, messages, config):
        return self.before_model(messages, config)

    async def aafter_model(self, ai_message, config):
        return self.after_model(ai_message, config)

# Usage:
agent = create_agent(
    model=llm, tools=tools,
    middleware=[TokenBudgetMiddleware(max_tokens_per_session=50_000)],
)
```

### All `AgentMiddleware` hook signatures

```python
class AgentMiddleware:
    # Entry/exit of the full agent run:
    def before_agent(self, state: dict, config: RunnableConfig) -> dict: ...
    def after_agent(self, state: dict, config: RunnableConfig) -> dict: ...

    # Called before/after each model invocation:
    def before_model(self, messages: list[BaseMessage], config) -> list[BaseMessage]: ...
    def after_model(self, ai_message: AIMessage, config) -> AIMessage: ...

    # Wraps the model call — intercept to implement caching, routing, etc.:
    def wrap_model_call(self, messages, call_next: Callable, config) -> AIMessage: ...
    # call_next(messages) executes the next middleware's wrap_model_call or the actual model.

    # Wraps each tool call:
    def wrap_tool_call(self, tool_name: str, tool_args: dict, call_next: Callable, config) -> Any: ...

    # Async equivalents (optional — sync versions used as fallback):
    async def abefore_agent(self, state, config): ...
    async def aafter_agent(self, state, config): ...
    async def abefore_model(self, messages, config): ...
    async def aafter_model(self, ai_message, config): ...
    async def awrap_model_call(self, messages, call_next, config): ...
    async def awrap_tool_call(self, tool_name, tool_args, call_next, config): ...
```

---

## Middleware vs LangGraph nodes

| Concern | Use middleware | Use LangGraph nodes |
|---|---|---|
| Lifecycle hooks (logging, limits, PII) | ✓ | — |
| Graceful context management | ✓ (SummarizationMiddleware) | — |
| HITL approval of tool calls | ✓ (HumanInTheLoopMiddleware) | — if you need richer control |
| Multi-step orchestration | — | ✓ |
| Parallel subagents | — | ✓ (Send API) |
| Conditional branching between agents | — | ✓ |
| Plan-execute-reflect loops | — | ✓ |
| Custom state management with reducers | — | ✓ |

For simple agent customization, middleware is the right level. For complex multi-step or multi-agent workflows, drop to `StateGraph` directly.

---

## Gotchas

1. **`before_model` and `after_model` must always return** their argument (or a modified version). Returning `None` crashes the agent loop.

2. **`wrap_model_call` must call `call_next`** or return a synthetic `AIMessage`. Not calling `call_next` effectively short-circuits the model call — useful for caching but dangerous if misused.

3. **`HumanInTheLoopMiddleware` requires a checkpointer.** Without a checkpointer, interrupts can't be persisted and resuming won't work.

4. **Middleware order is stable across turns.** The list passed to `create_agent` is fixed for the lifetime of that agent instance. If you need per-invocation middleware configuration, use `context_schema` + conditional logic inside middleware hooks.

5. **Async performance:** if your middleware hooks do I/O (database writes, API calls), implement the `async` variants (`abefore_model`, etc.) to avoid blocking the async event loop.
