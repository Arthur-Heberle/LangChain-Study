from dotenv import load_dotenv
import os
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_tavily import TavilySearch
from langchain.agents import create_agent

load_dotenv()

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    temperature=0
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001", 
    google_api_key=os.getenv("GEMINI_API_KEY")
)

vectorstore = Chroma(
    persist_directory="/home/aheberle/study/langchain/4-rag/chroma_db",
    embedding_function=embeddings
)

# Tool that encapsulates RAG
@tool
def search_immoveable(question: str) -> str:
    """
    Searches the JR Imóveis internal database for specific property information.
    ALWAYS use this tool first when the user asks about:
    - Properties available for sale or rent
    - Prices, sizes, locations, or features of specific properties
    - JR Imóveis policies, documentation requirements, or processes
    - Neighborhoods in Catanduvas, SC
    Only use web search if this tool returns no relevant results.

    Args:
        question: The search query in natural language
    """

    results = vectorstore.similarity_search_with_score(question, k=4)
    
    chunks_relevantes = [
        doc for doc, score in results if score < 0.8
    ]

    if not chunks_relevantes:
        return "No information found in the JR Imóveis database for this query."

    # Passa só o page_content de cada chunk relevante
    return "\n\n".join(doc.page_content for doc in chunks_relevantes)

@tool
def calculate_financing(imovel_value: float, percentage_down_payment: float, years: int) -> str:
    """
    Calculates the monthly installment for a real estate financing.
    Use this tool when the user asks about financing, monthly payments,
    or installment values for a property purchase.

    Args:
        imovel_value: Total property value in BRL
        percentage_down_payment: Down payment percentage (0 to 100)
        years: Financing term in years
    """    
    
    monthly_fee = 0.009 # 10.8 per year
    down_payment = imovel_value * (percentage_down_payment/100)
    financed = imovel_value - down_payment
    months = years * 12

    installment = financed * (monthly_fee * (1 + monthly_fee) ** months) / ((1 + monthly_fee) ** months - 1)

    return (
        f"Down payment: R$ {down_payment:,.2f}\n"
        f"Financed amount: R$ {financed:,.2f}\n"
        f"Estimated monthly payment: R$ {installment:,.2f}"
    )

search_web = TavilySearch(max_results=3)
tools = [search_immoveable, calculate_financing, search_web]

agent = create_agent(llm, tools)

print("=== AGENT JR IMÓVEIS — RAG + Web + Calculations ===")
print("Type 'exit' to end conversation\n")

while True:
    question = input("You: ").strip()
    if question.lower() == "exit":
        break
    if not question:
        continue

    result = agent.invoke({
        "messages": [{"role": "user", "content": question}]
    })

    # Imprime TODAS as mensagens para você ver quais tools foram chamadas
    print("\n--- TOOL CALLS ---")
    for msg in result["messages"]:
        tipo = type(msg).__name__
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                print(f"Tool chamada: {tc['name']} | Args: {tc['args']}")
        elif tipo == "ToolMessage":
            print(f"Tool retornou: {msg.content[:200]}")
    print("--- FIM TOOL CALLS ---\n")

    final_result = result["messages"][-1].content
    print(f"Assistant: {final_result}\n")
    print("="*50 + "\n")
