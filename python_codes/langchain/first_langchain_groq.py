import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

# 1. Load the environment variables from the .env file
load_dotenv()

# 2. Fetch the variable from the system environment
#api_key = os.getenv("GROQ_API_KEY")
#print(api_key)

llm = ChatGroq(
    model="qwen/qwen3-32b",
    temperature=0,
    max_tokens=None,
    reasoning_format="parsed",
    timeout=None,
    max_retries=2,
    # other params...
)


messages = [
    (
        "system",
        "I want to open a restaurant in based on country food. suggest a fancy name for this",
    ),
    ("human", "India"),
]

ai_msg = llm.invoke(messages)
print(ai_msg.content)

