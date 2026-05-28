from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    temperature=0.2
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001", 
    google_api_key=os.getenv("GEMINI_API_KEY")
)

vectorstore = Chroma(
    persist_directory="./4-rag/chroma_db",
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Prompt RAG - "use ONLY the context below".
# {context} recieve the chunks from reciever .
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a real estate assistant for JR Imóveis in Catanduvas, SC. "
     "Answer the user's question using ONLY the context provided below. "
     "If the answer is not in the context, say: 'I don't have that information in my database.' "
     "Do not invent information. Plain text only, no markdown, no asterisks.\n\n"
     "Context:\n{context}"),
    ("human", "{question}")
])

parser = StrOutputParser()

# RunnablePassthrough pass the question without modify it.
# Guarantee that retriever and prompt get the same question

# {"question": "..."} 
# RunnableParallel 2 paths:
#   - "context": retriever recieve question, return chunks
#   - "question": RunnablePassthrough pass the question
# the result is a dict {"context": "...", "question": "..."}
# to the prompt

def format_docs(docs):
    """return chunks in a single text separated by 2 new lines."""
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | parser
)

print("=== RAG - JR Imóveis ===")
print("Type 'exit' to end conversationr\n")

while True:
    question = input("Você: ").strip()
    if question.lower() == "sair":
        break
    if not question:
        continue

    resposta = rag_chain.invoke(question)
    print(f"Assistente: {resposta}\n")