# 02 — Prompts and Templates

All prompt classes live in `langchain_core.prompts`. Every prompt template is a `Runnable` and can be composed with `|`.

---

## `PromptTemplate` — string prompts

For non-chat models or when you want a single formatted string output.

```python
from langchain_core.prompts import PromptTemplate

# From a template string with {variables}
prompt = PromptTemplate.from_template("Translate '{text}' to {language}.")

# Explicit (preferred when you want input_variables documented)
prompt = PromptTemplate(
    template="Translate '{text}' to {language}.",
    input_variables=["text", "language"],
)

# Partial variables — pre-fill some inputs at construction time
prompt = PromptTemplate(
    template="As of {date}, the status is: {status}",
    input_variables=["status"],
    partial_variables={"date": "2026-04-25"},
)

# Invoke returns a StringPromptValue
value = prompt.invoke({"text": "hello", "language": "French"})
print(value.to_string())   # "Translate 'hello' to French."
```

---

## `ChatPromptTemplate` — multi-message prompts (the default for chat models)

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Tuple shorthand: (role, content_template)
# Roles: "system", "human", "ai", "tool"
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a {persona} expert. Respond in {language}."),
    ("human", "{question}"),
])
result = prompt.invoke({"persona": "Python", "language": "English",
                        "question": "What is a decorator?"})
# result is a ChatPromptValue containing a list of BaseMessage objects

# Use | to chain with an LLM
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser

chain = prompt | ChatGoogleGenerativeAI(model="gemini-2.5-flash") | StrOutputParser()
chain.invoke({"persona": "Python", "language": "English", "question": "What is a decorator?"})
```

### `from_messages` input formats

All three forms are equivalent and can be mixed in a single list:

```python
from langchain_core.messages import SystemMessage, HumanMessage

ChatPromptTemplate.from_messages([
    # Tuple form (most common):
    ("system", "You are {persona}."),

    # Message object form:
    SystemMessage(content="You are {persona}."),    # NOTE: {} templating NOT applied to
                                                     # pre-built message objects unless you
                                                     # use PromptTemplate inside them.

    # Template + message object combo (use HumanMessagePromptTemplate for templated human):
    ("human", "{question}"),
])
```

**Gotcha:** `MessageObject(content="...")` inside `from_messages` does NOT apply `{}` variable substitution. Only tuple strings do. Use `("human", "{variable}")` tuples for dynamic content.

---

## `MessagesPlaceholder` — injecting message lists at runtime

Essential for conversation history and agent scratchpads.

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="history", optional=True),  # optional=True means
                                                                   # the key can be absent
    ("human", "{question}"),
])

# Pass a list of messages (e.g., from checkpointer or external store):
from langchain_core.messages import HumanMessage, AIMessage
response = (prompt | llm).invoke({
    "history": [
        HumanMessage(content="Hi, I'm Alice."),
        AIMessage(content="Hello Alice! How can I help?"),
    ],
    "question": "What did I just tell you my name was?",
})
```

**Tuple shorthand equivalent:**
```python
# ("placeholder", "{variable_name}") is identical to MessagesPlaceholder("variable_name")
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are helpful."),
    ("placeholder", "{history}"),    # same as MessagesPlaceholder("history")
    ("human", "{question}"),
])
```

---

## `FewShotPromptTemplate` — few-shot for string/completion prompts

```python
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate

examples = [
    {"input": "happy",   "output": "sad"},
    {"input": "tall",    "output": "short"},
    {"input": "energetic","output": "lethargic"},
]

example_prompt = PromptTemplate.from_template("Input: {input}\nOutput: {output}")

fs_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="Give the antonym of every input.",
    suffix="Input: {adjective}\nOutput:",
    input_variables=["adjective"],
    example_separator="\n\n",
)
fs_prompt.invoke({"adjective": "joyful"})
```

---

## `FewShotChatMessagePromptTemplate` — few-shot for chat models

```python
from langchain_core.prompts import (
    ChatPromptTemplate, FewShotChatMessagePromptTemplate,
)

examples = [
    {"input": "2+2",    "output": "4"},
    {"input": "2+3",    "output": "5"},
]

example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{input}"), ("ai", "{output}"),
])

few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
    input_variables=["input"],
)

# Embed inside a larger prompt:
final_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a math assistant."),
    few_shot_prompt,                  # expands to the example messages
    ("human", "{input}"),
])
```

---

## `SemanticSimilarityExampleSelector` — dynamic few-shot

Choose the most relevant examples per query based on embedding similarity:

```python
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

examples = [
    {"input": "pirate", "output": "ship"},
    {"input": "pilot",  "output": "plane"},
    {"input": "driver", "output": "car"},
    {"input": "tree",   "output": "ground"},
    {"input": "bird",   "output": "nest"},
]

selector = SemanticSimilarityExampleSelector.from_examples(
    examples,
    GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001", task_type="SEMANTIC_SIMILARITY"
    ),
    Chroma,
    k=2,                    # number of examples to select
    input_keys=["input"],   # keys used for similarity scoring
)

fs_prompt = FewShotChatMessagePromptTemplate(
    example_selector=selector,      # pass selector instead of examples
    example_prompt=example_prompt,
    input_variables=["input"],
)
```

Other selectors: `LengthBasedExampleSelector` (controls token budget), `MaxMarginalRelevanceExampleSelector` (MMR for diversity).

---

## Partial variables

Pre-bind values to a prompt so callers don't need to pass them:

```python
from datetime import datetime

prompt = ChatPromptTemplate.from_messages([
    ("system", "The current date is {date}. You are a {persona}."),
    ("human", "{question}"),
])

# Static partial:
prompt_with_date = prompt.partial(date="2026-04-25")

# Dynamic partial (callable, evaluated at invocation time):
prompt_dynamic = prompt.partial(date=lambda: datetime.now().strftime("%Y-%m-%d"))
prompt_dynamic.invoke({"persona": "assistant", "question": "What day is today?"})
```

---

## Composing prompts with LCEL

```python
# Prompt as a Runnable step
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser

chain = prompt | ChatGoogleGenerativeAI(model="gemini-2.5-flash") | StrOutputParser()
chain.invoke({"question": "What is LCEL?", "history": []})
```

---

## `SystemMessagePromptTemplate` / `HumanMessagePromptTemplate` (rarely needed)

These wrapper classes let you use template variables inside message objects. In practice, the `("system", "{var}")` tuple syntax in `ChatPromptTemplate.from_messages` is simpler and fully equivalent.

```python
from langchain_core.prompts.chat import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

# Only necessary if you need the PromptTemplate object for something specific
sys = SystemMessagePromptTemplate.from_template("You are {persona}.")
human = HumanMessagePromptTemplate.from_template("{question}")
```

---

## Gotchas

1. **Message objects in `from_messages` are NOT templated.** `SystemMessage(content="You are {persona}.")` will pass the literal string with braces. Use tuple form `("system", "You are {persona}.")` for template variables.

2. **`optional=True` on `MessagesPlaceholder`** prevents KeyError when the key is absent. Set it when history is optional (e.g., first turn of a conversation).

3. **`input_variables` must be exhaustive.** If a template references `{foo}` and `foo` isn't in `input_variables`, invocation raises a `KeyError`. Use `partial_variables` to pre-bind values known at construction time.

4. **LangGraph vs manual prompt assembly.** When building a LangGraph agent, you generally construct prompts inside node functions rather than as standalone templates — the state dict is the input. Reserve `ChatPromptTemplate` chains for non-agent LCEL pipelines.

5. **System prompt on `create_agent`.** Pass `system_prompt="..."` to `create_agent` — it's simpler than manually building a `ChatPromptTemplate` just for the system message.
