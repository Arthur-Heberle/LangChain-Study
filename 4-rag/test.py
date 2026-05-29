from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
import os
from dotenv import load_dotenv
load_dotenv()
embeddings = GoogleGenerativeAIEmbeddings(model='models/gemini-embedding-001', google_api_key=os.getenv('GEMINI_API_KEY'))
db = Chroma(persist_directory='./4-rag/chroma_db', embedding_function=embeddings)
docs = db.get()
print(f'Total de chunks: {len(docs)}')
for i, doc in enumerate(docs["documents"]):
    print(f'\\nChunk {i}: {doc[:250]}')