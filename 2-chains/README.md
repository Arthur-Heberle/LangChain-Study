# Day 2 — Chains

Chaining multiple processing steps with LangChain. The output of one chain becomes the input of the next, enabling complex pipelines in a readable way.

## Concepts covered

- What a `Runnable` is and why every LangChain component implements this interface
- How to pass the output of one chain into another using a `lambda`
- `RunnableParallel` — running multiple chains simultaneously
- Cost trade-off: every `|` that passes through an LLM is an API call
- When to use sequential pipelines vs parallel execution

## Files

| File | Description |
|---|---|
| `sequential_chain.py` | Two chains in sequence: analyzes a property → generates a client report |
| `parallel_chain.py` | Two chains in parallel: pros and risks simultaneously → final assessment |
| `exercicio.py` | Three-step pipeline: neighborhood profile → price range → ideal tenant profile |

## How to run

```bash
source .venv/bin/activate

python 2-chains/sequential_chain.py
python 2-chains/parallel_chain.py
python 2-chains/exercise.py
```

## Required environment variables

```
DEEPSEEK_API_KEY=sk-...
```

## Pattern for connecting chains

```python
# The output of chain_a is a string.
# The lambda wraps that string into the dictionary chain_b expects.
pipeline = chain_a | (lambda output: {"key": output}) | chain_b
```

## What you should understand by the end of this module

- Why the lambda is necessary between two sequential chains
- When to use `RunnableParallel` and what the extra token cost implies
- How LangChain knows how to connect components through the `|` operator