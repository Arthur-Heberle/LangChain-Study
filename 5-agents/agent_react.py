from dotenv import load_dotenv
import os
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain.agents import create_agent

load_dotenv()

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    temperature=0  # 0 is essencial in agents deterministic responses
)

# Web search tool, Tavily returns clean results for LLMs
web_search = TavilySearch(
    max_results=3,
    description=(
        "Search the web for current information about real estate markets, "
        "property prices, interest rates, or any topic requiring up-to-date data."
    )
)

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

@tool
def calculate_rentability(property_value: float, monthly_rent: float) -> str:
    """
    Calculates the annual rental yield of a property.
    Use when the user asks about rental return, yield, or investment viability.

    Args:
        property_value: Property purchase price in BRL
        monthly_rent: Expected monthly rent in BRL
    """
    rental_yield_annual = (monthly_rent * 12 / property_value) * 100
    return (
        f"Annual rental yield: {rental_yield_annual:.2f}%\n"
        f"Monthly: R$ {monthly_rent:,.2f} | Annual: R$ {monthly_rent*12:,.2f}\n"
        f"Reference: Brazilian real estate average is 4-6% per year."
    )

tools = [web_search, calculate_financing, calculate_rentability]

agent = create_agent(llm, tools)

print("=== REAL ESTATE AGENT ===")
print("Type 'exit' to end conversation\n")

while True:
    question = input("You: ").strip()
    if question.lower() == "exit":
        break
    if not question:
        continue

    # The LangGraph agent receives a dictionary with "messages"
    result = agent.invoke({
        "messages": [{"role": "user", "content": question}]
    })

    # The final answer is the last message in the history
    final_result = result["messages"][-1].content
    print(f"\nAssistant: {final_result}\n")
    print("="*50 + "\n")