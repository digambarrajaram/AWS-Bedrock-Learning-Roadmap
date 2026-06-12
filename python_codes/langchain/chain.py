import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

# 1. Load the environment variables from the .env file
load_dotenv()

llm = ChatGroq(
    model="qwen/qwen3-32b",
    temperature=0,
    max_tokens=None,
    reasoning_format="parsed",
    timeout=None,
    max_retries=2,
    # other params...
)


prompt = PromptTemplate.from_template(
    "I want to open a restaurant for {cuisine} food. suggest a fancy name for this "
)

# Chains are linked directly using pipes
chain = prompt | llm

# You execute the chain using .invoke() instead of .run()
response = chain.invoke({"cuisine": "Indian"})
print(response.content)