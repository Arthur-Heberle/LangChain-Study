from dotenv import load_dotenv
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001", 
    google_api_key=os.getenv("GEMINI_API_KEY")
)

vectorstore = Chroma(
    persist_directory="./4-rag/chroma_db",
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever( # Transforms vectorstore in Runnable
    search_type="similarity",  # standard — others: "mmr", "similarity_score_threshold"
    search_kwargs={"k": 3}  
)

docs = retriever.invoke("qual o preço das casas disponíveis?")

print("=== BASIC RETRIEVER ===")
print(f"Query: 'qual o preço das casas disponíveis?'")
print(f"Chunks returned: {len(docs)}\n")
for i, doc in enumerate(docs):
    print(f"--- Chunk {i+1} ---")
    print(doc.page_content)
    print()

# MMR = Maximal Marginal Relevance

retriever_mmr = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3,
        "fetch_k": 10,    # search 10, select 3
        "lambda_mult": 0.7 # 0=max diversity, 1=max relevance
    }
)

docs_mmr = retriever_mmr.invoke("imóveis disponíveis para comprar")
print("=== RETRIEVER MMR (diversity) ===")
print(f"Query: 'imóveis disponíveis para comprar'")
for i, doc in enumerate(docs_mmr):
    print(f"--- Chunk {i+1} ---")
    print(doc.page_content[:200])
    print()