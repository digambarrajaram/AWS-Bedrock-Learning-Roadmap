from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from typing import TypedDict,Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()
memory = MemorySaver()

config = {'configurable': {'thread_id': '1'}}

class State(TypedDict):
    messages: Annotated[list, add_messages]

@tool
def get_stock_price(symbol: str) -> float:
    """Fetch a stock price for the given ticker symbol."""
    # Placeholder implementation - replace with actual stock price fetching logic
    return {
        "MSFT": 100.0,
        "AAPL": 150.0,
        "GOOGL": 200.0,
        "AMZN": 1000.0,
        "RIL": 50.0
    }.get(symbol, 0.0)

tools = [get_stock_price]

llm = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0.7,
    max_tokens=1024
)

llm_with_tools = llm.bind_tools(tools)

def chatbot(state: State) -> State:
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


builder = StateGraph(State)

builder.add_node("chatbot", chatbot)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "chatbot")
builder.add_conditional_edges("chatbot", tools_condition)
builder.add_edge("tools", "chatbot")
builder.add_edge("chatbot", END)    

tool_bot = builder.compile(checkpointer=memory)

response = tool_bot.invoke({"messages": [{"role": "user", "content": "what is addition of two stock prices of MSFT and AAPL?"}]}, config=config)
print(response["messages"][-1].content)
print("-------------------------------------------------------------------\n")
response = tool_bot.invoke({"messages": [{"role": "user", "content": "add a stock price of GOOGL to previous total stock price"}]}, config=config)
print(response["messages"][-1].content)

