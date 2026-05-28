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
    temperature = 0.3 # 0 = determinístico, 1 = criativo
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a professional consultant in the real estate market, specialized in this city: {city}."),
    ("human", "{question}" )
])

parser = StrOutputParser()

chain = prompt | llm | parser

imovel_type = "apartment"
city = "Curitiba, PR, Brazil"
goal = "buy"

question_input = f"Where are the best neighborhoods to {goal} an {imovel_type} in {city}?"

response = chain.invoke({
    "city": city,
    "question": question_input

})


print(response.replace("*",""))