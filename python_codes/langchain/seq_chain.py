import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

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


prompt_name = PromptTemplate.from_template(
    "I want to open a restaurant for {cuisine} food. suggest only one fancy name for this "
)

# The variable name here MUST match what the previous step outputs (e.g., 'restaurant_name')
prompt_menu = PromptTemplate.from_template(
    "Suggest a 3-item menu for a restaurant named: {restaurant_name}"
)



# Build the Sequential Chain using Pipes (|)
# Note: StrOutputParser extracts just the text string from the LLM response
modern_sequential_chain = (
    {"restaurant_name": prompt_name | llm | StrOutputParser()} 
    | prompt_menu 
    | llm
)

# You execute the chain using .invoke() instead of .run()
response = modern_sequential_chain.invoke({"cuisine": "Maharashtiran"})
print(response.content)