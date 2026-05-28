from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

load_dotenv()


llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    temperature=0.3
)

parser = StrOutputParser()

# Chain A: analyze positive aspects
prompt_pros = ChatPromptTemplate.from_messages([
    ("system", "You are a real estate expert. List only the POSITIVE aspects. Plain text, no markdown."),
    ("human", "Property: {type} in {neighborhood}, {city}. {area}m2 for R${price}.")
])

# Chain B: analyze risks
prompt_cons = ChatPromptTemplate.from_messages([
    ("system", "You are a real estate risk analyst. List only the RISKS and CONCERNS. Plain text, no markdown."),
    ("human", "Property: {type} in {neighborhood}, {city}. {area}m2 for R${price}.")
])

chain_pros = prompt_pros | llm | parser
chain_cons = prompt_cons | llm | parser

parallel_analysis = RunnableParallel({
    "positive_aspects": chain_pros,
    "risks": chain_cons
})

# Final Chain: union both chains
prompt_union = ChatPromptTemplate.from_messages([
    ("system", "You are a senior real estate consultant. Write a balanced final opinion."),
    ("human",
     "Positive aspects:\n{positive_aspects}\n\n"
     "Risks:\n{risks}\n\n"
     "Write a balanced final recommendation.")
])

final_chain = prompt_union | llm | parser

# Complete Pipeline : parallel → final chain
pipeline = parallel_analysis | final_chain

result = pipeline.invoke({
    "type": "house",
    "neighborhood": "Batel",
    "city": "Curitiba",
    "area": "200",
    "price": "2500000"
})

print("=== PARECER FINAL ===")
print(result)