from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    temperature=0.4
)

parser = StrOutputParser()

# Chain 1: imovel analysis
prompt_analysis = ChatPromptTemplate.from_messages([
    ("system", 
     "You are a real estate analyst. Analyze the property objectively."),
    ("human", 
     "Property: {type}, located in {neighborhood}, {city}"
     "Area: {area}m2. Price: R${price}."
     "Provide a technical analysis of this property.")
])

chain_analysis = prompt_analysis | llm | parser


# Chain 2: Create a report with the analysis
prompt_report = ChatPromptTemplate.from_messages([
    ("system",
     "You are a real estate agent writing to a potential buyer. "
     "Transform the technical analysis into a friendly, persuasive client report. "
     "Plain text only, no markdown, no asterisks."),
    ("human",
     "Technical analysis:\n{analysis}\n\n"
     "Write a client-friendly report based on this analysis.")
])

chain_report = prompt_report | llm | parser

# Conecting chains
from langchain_core.runnables import RunnablePassthrough

# lambda takes chain_analysis response as String, and make a dict with it in the variable "analysis"
chain = chain_analysis | (lambda analysis: {"analysis": analysis}) | chain_report

# Execution
result = chain.invoke({
    "type": "apartment",
    "neighborhood": "Água Verde",
    "city": "Curitiba",
    "area": "85",
    "price": "650000"
})

print("==== Report ====")
print(result)