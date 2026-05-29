from dotenv import load_dotenv
import os
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

load_dotenv()

# @tool transforms python func into a langchain tool
# LLM reads docstring to decide when to use the tool
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
    down_payment = imovel_value * (percentage_down_paymment/100)
    financed = imovel_value - down_payment
    months = years * 12

    installment = financed * (monthly_fee * (1 + monthly_fee) ** months) / ((1 + monthly_fee) ** months - 1)

    return (
        f"Property value: R$ {imovel_value:,.2f}\n"
        f"Down payment ({percentage_down_paymment}%): R$ {down_payment:,.2f}\n"
        f"Financed amount: R$ {financed:,.2f}\n"
        f"Term: {years} years ({months} months)\n"
        f"Estimated monthly payment: R$ {installment:,.2f}\n"
        f"Note: Rate used: 0.9%/month (~10.8%/year). Actual rate varies by bank."
    )

@tool
def convert_area(value: float, from_value: str, to_value: str) -> str:
    """
    Converts area measurements between units.
    Use when the user asks to convert m², hectares, or square feet.

    Args:
        value: Numeric value to convert
        from_value: Source unit (m2, hectare, sqft)
        to_value: Target unit (m2, hectare, sqft)
    """
    conversions = {
        ("m2", "hectare"): 0.0001,
        ("hectare", "m2"): 10000,
        ("m2", "sqft"): 10.7639,
        ("sqft", "m2"): 0.092903,
        ("hectare", "sqft"): 107639,
        ("sqft", "hectare"): 0.0000092903
    }
    key = (from_value.lower(), to_value.lower())
    if key not in conversions:
        return f"Conversion from {from_value} to {to_value} not supported."
    result = value * conversions[key]
    return f"{value} {from_value} = {result:.4f} {to_value}"

# Inspecionando uma tool — veja o que o LLM enxerga
print("=== WHAT THE LLM SEES ABOUT THE TOOL ===")
print(f"Name: {calculate_financing.name}")
print(f"Description:\n{calculate_financing.description}")
print(f"Args: {calculate_financing.args}")

