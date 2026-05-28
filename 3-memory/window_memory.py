from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnableLambda

load_dotenv()

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    temperature=0.5
)

parser = StrOutputParser()

WINDOW_SIZE = 6  # stores 6 messages (3 turns)

def get_windowHistory(session_id: str, store: dict) -> ChatMessageHistory:
    if session_id not in store: 
        store[session_id] = ChatMessageHistory()
    history = store[session_id]

    while len(history.messages) > WINDOW_SIZE:
        history.messages.pop(0)
    return history

prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a real estate assistant. "
     "You only remember the last few messages. "
     "Plain text, no markdown, no asterisks."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

chain = prompt | llm | parser
store = {}

chain_window = RunnableWithMessageHistory(
    chain,
    lambda session_id: get_windowHistory(session_id, store),
    input_messages_key="input",
    history_messages_key="chat_history"
)

config = {"configurable": {"session_id": "test_window"}}

print("=== CHATBOT WINDOW MEMORY (3 turns) ===")
print("Type 'exit' to finish conversation.")

while True:
    user_input = input("You: ").strip()
    if user_input.lower() == "exit":
        print("Ending conversation.")
        break

    if not user_input:
        continue

    response = chain_window.invoke(
        {"input": user_input},
        config=config
    )
    session = store.get("test_window")
    total = len(session.messages) if session else 0
    print(f"Assistant: {response}\n")
    print(f"[current history: {total} messages]")