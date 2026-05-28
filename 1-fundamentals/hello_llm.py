# Carregamos a biblioteca que lê o arquivo .env
# O .env guarda sua chave de API sem expô-la no código
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI

load_dotenv()  # Lê o .env e joga as variáveis no ambiente do processo

llm = ChatOpenAI(
    model = "deepseek-chat",
    api_key = os.getenv("DEEPSEEK_API_KEY_TOSTUDY"),
    base_url = "https://api.deepseek.com",
    temperature = 0.7 # 0 = determinístico, 1 = criativo
)

response = llm.invoke("como falar olá mundo em ingles?")

print(f"""{response.content} \n--- \n {type(response)}""")