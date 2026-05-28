from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()

parser = StrOutputParser()

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    temperature=0.5
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a real estate assistant specialized in Brazil. "
     "Plain text, no markdown, no asterisks."),
    MessagesPlaceholder(variable_name="chat_history"),  # historic
    ("human", "{input}")                                # message
])

chain = prompt | llm | parser

# Dictionary that stores histories by session_id.
# RAM Memory - would be a database.
histories = {}

def get_history(session_id: str) -> ChatMessageHistory:
    if session_id not in histories:
        histories[session_id] = ChatMessageHistory()
    return histories[session_id]

chain_with_memory = RunnableWithMessageHistory(
    chain,
    get_history,
    input_messages_key="input",
    history_messages_key="chat_history"
)

config = {"configurable": {"session_id": "chat"}}

print("=== CHATBOT REAL ESTATE MARKET ===")
print("Type 'exit' to finish conversation.")

while True:
    user_input = input("You: ").strip()

    if user_input.lower() == "exit":
        print("Ending conversation.")
        break

    if not user_input:
        continue

    response = chain_with_memory.invoke(
        {"input": user_input},
        config=config
    )
    print(f"Assistant: {response}\n")