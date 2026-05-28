from dotenv import load_dotenv
import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

load_dotenv()

from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",  # Best model to embedding on gemini
    google_api_key=os.getenv("GEMINI_API_KEY")
)

loader = TextLoader("4-rag/documents/immoveable.txt", encoding="utf-8")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
chunks = splitter.split_documents(docs)

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./4-rag/chroma_db"  # persiste em disco
)

print(f"Complete indexing. {vectorstore._collection.count()} vectors in base\n")

print("=== SIMILARITY SEARCH TEST ===\n")
query1 = "apartamento para alugar"
results1 = vectorstore.similarity_search(query1,k=2)

for i, doc in enumerate(results1):
    print(f"Result {i+1}: {doc.page_content[:200]}\n")

print("=== SIMILARITY SEARCH WITH SCORE TEST ===\n") # Score close to 0 is more similar 
results_score = vectorstore.similarity_search_with_score("terreno à venda", k=3)
for doc, score in results_score:
    print(f"Score: {score:.4f} | Chunk: \n{doc.page_content[:150]}\n")