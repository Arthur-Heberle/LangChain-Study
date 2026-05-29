# 3 — Memory & Conversation

Adding memory to a chatbot. The LLM has no native memory — the conversation history is accumulated in Python and resent on every call.

## Concepts covered

- Why LLMs have no memory and how it is simulated
- `ChatMessageHistory` — object that accumulates conversation messages
- `MessagesPlaceholder` — special placeholder that injects history into the prompt
- `RunnableWithMessageHistory` — wrapper that automates history injection and saving
- `session_id` — how to isolate multiple parallel conversations
- Window memory — limiting history to the last N turns to control cost

## Files

| File | Description |
|---|---|
| `chat_history.py` | Chatbot with full memory using `RunnableWithMessageHistory` |
| `window_memory.py` | Chatbot with window memory — keeps only the last N messages |
| `exercicio.py` | JR Imóveis chatbot with client name collected before the loop |

## How to run

```bash
source .venv/bin/activate

python 3-memory/01_chat_history.py
python 3-memory/02_window_memory.py
python 3-memory/exercicio.py
```

## Required environment variables

```
DEEPSEEK_API_KEY=sk-...
```

## Memory trade-offs

| Type | Advantage | Disadvantage |
|---|---|---|
| Full buffer | No detail is lost | Gets expensive with long conversations |
| Window (last N) | Controlled cost | Loses context from older turns |
| Summary | Compresses history | Loses specific details |

## How `session_id` works

The history is stored in a Python dictionary indexed by `session_id`. In production, this dictionary would be replaced by a database (Redis, PostgreSQL), but the mechanism is identical.

```python
# Multiple simultaneous users with isolated histories
config_a = {"configurable": {"session_id": "user_arthur"}}
config_b = {"configurable": {"session_id": "user_nicolly"}}
```

## What you should understand by the end of this module

- Why the LLM can "remember" even without internal state
- The difference between `MessagesPlaceholder` and a regular `{variable}` placeholder
- How `RunnableWithMessageHistory` intercepts the invoke before and after calling the LLM