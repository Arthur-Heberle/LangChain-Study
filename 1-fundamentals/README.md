# Day 1 — Fundamentals

First contact with LangChain: connecting to an LLM, using PromptTemplates, and building a basic chain with the pipe operator (`|`).

## Concepts covered

- What LangChain is and what it is used for
- How to connect to DeepSeek using `ChatOpenAI` with a custom `base_url`
- Difference between calling the LLM with a plain string vs a `PromptTemplate`
- What LCEL (LangChain Expression Language) is and how the `|` operator works
- What an `AIMessage` is and how to extract text with `.content`
- How `StrOutputParser` eliminates the need to call `.content` manually

## Files

| File | Description |
|---|---|
| `hello_llm.py` | First LLM call. Displays the raw `AIMessage` object |
| `prompt_template.py` | Uses `ChatPromptTemplate` with variables and builds the chain with `\|` |
| `exercise.py` | Chain with three variables: property type, city, and goal |

## How to run

```bash
# Activate the virtual environment from the project root
source .venv/bin/activate

# Run any file directly
python 1-fundamentals/hello_llm.py
python 1-fundamentals/prompt_template.py
python 1-fundamentals/exercicio.py
```

## Required environment variables

```
DEEPSEEK_API_KEY=sk-...
```

## What you should understand by the end of this module

- Why `chain = prompt | llm | parser` works
- The difference between `llm.invoke("text")` and using a `ChatPromptTemplate`
- Why environment variables exist and how `.env` works with `python-dotenv`