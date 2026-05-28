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

prompt_neighborhood = ChatPromptTemplate.from_messages([
    ("system",
     "Your role is to give me a feedback from a given neighborhood." 
     "I want to know the infrastructure, target public and security of the place."),
    ("human",
     "City: {city}"
     "Neighborhood: {neighborhood}")
])

chain_neighborhood = prompt_neighborhood | llm | parser

prompt_priceEstimate = ChatPromptTemplate.from_messages([
    ("system",
     "Your role is to estimate a price range for an imovel."),
    ("human",
     "Estimate a price for a rental with 2 bedrooms in this neighbor:\n {neighborhood_description}")
])

chain_price = prompt_priceEstimate | llm | parser

prompt_recomendation = ChatPromptTemplate.from_messages([
    ("system",
     "Your role is to create a recomendation about ideal tenants using a price range."),
    ("human",
     "The price range is: {price_range}")
])

chain_recommendation = prompt_recomendation | llm | parser

pipeline = (
    chain_neighborhood
    | (lambda desc: {"neighborhood_description": desc})
    | chain_price
    | (lambda price: {"price_range": price})
    | chain_recommendation
)

result = pipeline.invoke({
    "city": "Joaçaba",
    "neighborhood": "Flor da Serra",
})

print("===RESPONSE===")
print (result)
