# 4 — RAG (Retrieval Augmented Generation)

The most important module of the course. RAG allows the LLM to answer questions about documents it has never seen — without hallucinating — by retrieving only the relevant chunks and passing them as context.

## Concepts covered

- Why LLMs hallucinate when asked about unknown data
- What an embedding is and why semantically similar texts produce similar vectors
- What a vector store is and how similarity search works
- The two phases of RAG: indexing (once) and querying (every request)
- Document loaders — reading PDF, TXT, and CSV into `Document` objects
- Text splitters — chunking with overlap to preserve context across boundaries
- ChromaDB — local vector store with disk persistence
- Gemini `gemini-embedding-001` — production-grade embedding via API
- Retriever — the LangChain abstraction over the vector store used inside chains
- MMR (Maximal Marginal Relevance) — balancing relevance with diversity
- Full RAG chain with `RunnablePassthrough`
- RAG + Memory — combining document retrieval with conversation history
- Similarity scores — understanding how relevant a retrieved chunk actually is
- Common failure modes: bad chunking, hallucination when retriever finds nothing

## Files

| File | Description |
|---|---|
| `loaders.py` | Loads TXT and CSV files into `Document` objects. Demonstrates the difference in output between each loader type |
| `splitters.py` | Splits documents into chunks using `RecursiveCharacterTextSplitter`. Shows how `chunk_size` and `chunk_overlap` affect the result |
| `embeddings_vectorstore.py` | Indexes chunks into ChromaDB using Gemini embeddings. Demonstrates semantic search and similarity scores |
| `retriever.py` | Loads the existing ChromaDB and creates a retriever. Compares standard similarity search with MMR |
| `rag_chain.py` | Full RAG pipeline: question → retriever → context → LLM → answer. Runs as an interactive chatbot |
| `rag_with_memory.py` | RAG + conversation memory. The assistant retrieves documents and remembers previous turns |
| `exercicio.py` | RAG chatbot for JR Imóveis built from scratch |

## Documents used

```
4-rag/documents/
├── immoveable.txt   # property catalog and FAQ for JR Imóveis
└── immoveable.csv   # structured property listing
```

## How to run

```bash
source .venv/bin/activate

# Run embeddings_vectorstore.py first — it creates the ChromaDB database
python 4-rag/embeddings_vectorstore.py
python 4-rag/loaders.py
python 4-rag/splitters.py
python 4-rag/retriever.py
python 4-rag/rag_chain.py
python 4-rag/rag_with_memory.py
```

## Required environment variables

```
DEEPSEEK_API_KEY=sk-...
GOOGLE_API_KEY=...            # used for Gemini embeddings
```

## RAG pipeline at a glance

```
INDEXING PHASE (runs once)
Documents → TextSplitter → chunks → Gemini Embeddings → ChromaDB

QUERY PHASE (runs on every question)
Question → Gemini Embeddings → ChromaDB similarity search → top-K chunks
→ injected as {context} in prompt → LLM → answer grounded in documents
```

## Why Gemini for embeddings

DeepSeek does not expose an embedding API. Gemini text-embedding-004 is free up to 1500 requests/day, production-grade, and requires no local model download — making it the closest option to a real production setup.

## Key implementation pattern

```python
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | parser
)
```

## What you should understand by the end of this module

- Why RAG exists and what problem it solves compared to fine-tuning
- Why chunk size and overlap matter and how to tune them
- The difference between a vector store and a retriever in LangChain
- Why `RunnablePassthrough` is needed in the RAG chain
- How to force the LLM to not hallucinate beyond the retrieved context
- How MMR differs from standard similarity search