from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from typing import TypedDict,Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition


load_dotenv()



#response = llm.invoke("Who was the first person to walk on the moon?")
#print(response.content)


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
    return {"messages": [llm_with_tools.invoke(state["messages"]) ]}


builder = StateGraph(State)

builder.add_node("chatbot", chatbot)
builder.add_node("tools", ToolNode(tools, messages_key="messages"))

builder.add_edge(START, "chatbot")
builder.add_conditional_edges("chatbot", tools_condition)
builder.add_edge("tools", "chatbot")
builder.add_edge("chatbot", END)    

tool_bot = builder.compile()

#message = {"role": "user", "content": "What is the stock price of MSFT?"} #i wanted to buy 20 RIL stocks and 15 MSFT stocks then total how much profit i will get.

#response = tool_bot.invoke({"messages": [message]})
#print(response)

if __name__ == "__main__":
    state = None
    while True:
        in_msg = input("You: ")
        if in_msg.lower() == "quit" or in_msg.lower() == "exit":
            break
        if state is None:
            state = {"messages": [{"role": "user", "content": in_msg}]}
        else:
            state["messages"].append({"role": "user", "content": in_msg})

        response = tool_bot.invoke(state)
        print(f"Bot: {response['messages'][-1].content}")

