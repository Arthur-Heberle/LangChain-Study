# LangChain Intensive Course

A hands-on, one-week intensive study of LangChain focused on building real applications — not just copying code. Every module includes concept explanation, annotated examples, and a practical exercise built from scratch.

The project application throughout the course is a real estate assistant for **JR Imóveis**, a real estate agency in Catanduvas, SC, Brazil.

## Stack

| Tool | Purpose |
|---|---|
| Python 3.12 | Runtime |
| LangChain 0.3 | LLM orchestration framework |
| DeepSeek (`deepseek-chat`) | Main LLM — cheap and capable |
| Google Gemini (`text-embedding-004`) | Embedding model via API |
| ChromaDB | Local vector store for RAG |
| python-dotenv | Environment variable management |

## Project structure

```
langchain-curso/
├── .env                     # API keys — never commit this
├── .venv/                   # virtual environment
├── dia1_fundamentos/        # Day 1: LLMs, PromptTemplates, basic chains
├── dia2_chains/             # Day 2: Sequential and parallel chains
├── dia3_memory/             # Day 3: Conversation memory and session management
├── dia4_rag/                # Day 4: Retrieval Augmented Generation
├── dia5_agents/             # Day 5: Agents, Tools, and ReAct
├── dia6_langsmith/          # Day 6: Observability and debugging
└── dia7_projeto_final/      # Day 7: Full real estate assistant
```

## Setup

**1. Clone the repository and create the virtual environment**

```bash
git clone https://github.com/your-username/langchain-curso.git
cd langchain-curso

python3 -m venv .venv
source .venv/bin/activate
```

**2. Install dependencies**

```bash
pip install langchain langchain-openai langchain-community \
            langchain-chroma langchain-google-genai chromadb \
            python-dotenv faiss-cpu tiktoken tavily-python
```

**3. Configure environment variables**

Create a `.env` file in the project root:

```
DEEPSEEK_API_KEY=sk-...
GEMINI_API_KEY=...
```

- DeepSeek API key: [platform.deepseek.com](https://platform.deepseek.com)
- Google API key (free): [aistudio.google.com](https://aistudio.google.com)

**4. Run any module**

```bash
python 1-fundamentals/hello_llm.py
```

## Course modules

| Day | Module | Key concept |
|---|---|---|
| 1 | [Fundamentals](./1-fundamentals/README.md) | LLMs, PromptTemplates, LCEL pipe operator |
| 2 | [Chains](./2-chains/README.md) | Sequential and parallel chain composition |
| 3 | [Memory](./3-memory/README.md) | Conversation history and session management |
| 4 | RAG | Document loaders, embeddings, vector stores, retrieval |
| 5 | Agents | Tools, AgentExecutor, ReAct pattern |
| 6 | LangSmith | Tracing, observability, prompt optimization |
| 7 | Final project | Full real estate assistant with RAG + Memory + Agents |

## Design decisions

**Why DeepSeek instead of OpenAI?**
DeepSeek offers near-GPT-4 quality at a fraction of the cost (~$0.002/1M input tokens). LangChain uses the same `ChatOpenAI` class — only the `base_url` changes.

**Why Gemini for embeddings?**
DeepSeek does not expose an embedding API. Gemini's `text-embedding-004` model is free up to 1500 requests/day, production-grade, and requires no local model download.

**Why ChromaDB?**
Zero configuration, runs locally, persists to disk. The same concepts apply to production vector stores (Pinecone, Weaviate, pgvector).

## Key concepts at a glance

**LCEL pipe operator**
```python
chain = prompt | llm | parser
# prompt formats the input → llm generates a response → parser extracts the text
```

**RAG pipeline**
```python
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | parser
)
```

**Memory with session isolation**
```python
chain_with_memory = RunnableWithMessageHistory(
    chain,
    get_history,                          # function that returns history by session_id
    input_messages_key="input",
    history_messages_key="chat_history"
)
```