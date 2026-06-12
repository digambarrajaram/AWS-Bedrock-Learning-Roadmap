import os
from dotenv import load_dotenv
import warnings
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langgraph.prebuilt import create_react_agent
from langchain.agents import create_agent

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from langgraph.prebuilt import create_react_agent


load_dotenv()

def run_groq_tavily_agent():
    if not os.environ.get("GROQ_API_KEY"):
        raise ValueError("❌ Missing GROQ_API_KEY")
    if not os.environ.get("TAVILY_API_KEY"):
        raise ValueError("❌ Missing TAVILY_API_KEY")

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0
    )

    tools = [TavilySearch(max_results=3)]

    # No prompt needed — langgraph handles it internally
    agent = create_react_agent(
        model=llm,
        tools=tools
    )

    query_without_tool = "What was the highest closing stock price for Apple (AAPL) in December 2023?"
    print(f"\n🚀 Running query without tool: {query_without_tool}\n")
    response_without_tool = llm.invoke(query_without_tool)
    print("\n🚀 Running query without tool:\n")
    print(response_without_tool.content, "\n\n")

    

    query = "What was the highest closing stock price for Apple (AAPL) in December 2023?"
    print(f"\n🚀 Running query: {query}\n")

    response = agent.invoke({
        "messages": [{"role": "user", "content": query}]
    })

    print("\n✅ FINAL ANSWER:\n")
    print(response["messages"][-1].content)

if __name__ == "__main__":
    run_groq_tavily_agent()