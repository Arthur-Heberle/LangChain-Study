from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatOpenAI(
    model = "deepseek-chat",
    api_key = os.getenv("DEEPSEEK_API_KEY_TOSTUDY"),
    base_url = "https://api.deepseek.com",
    temperature = 0.7 # 0 = determinístico, 1 = criativo
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Você é um especialista em {area}. Responda de forma clara e direta."),
    ("human", "{pergunta}")
])

parser = StrOutputParser() # Pega o AIMessage e joga pra string

chain = prompt | llm | parser

areaBash = input("que area? ")
question = input("que pergunta? ")

response = chain.invoke({
    "area": areaBash,
    "pergunta": question
})

print(response)