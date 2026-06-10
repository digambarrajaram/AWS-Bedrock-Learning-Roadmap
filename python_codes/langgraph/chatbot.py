from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from typing import TypedDict,Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


load_dotenv()

llm = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0.7,
    max_tokens=1024
)
#response = llm.invoke("Who was the first person to walk on the moon?")
#print(response.content)


class State(TypedDict):
    msg: Annotated[list, add_messages]

def chatbot(state: State) -> State:
    return {"msg": [llm.invoke(state["msg"])]}


builder = StateGraph(State)

builder.add_node("chatbot", chatbot)

builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)    

bot = builder.compile()

message = {"role": "user", "content": "What is the capital of India?"}

#response = bot.invoke({"msg": [message]})
#print(response)

if __name__ == "__main__":
    state = None
    while True:
        in_msg = input("You: ")
        if in_msg.lower() == "quit" or in_msg.lower() == "exit":
            break
        if state is None:
            state = {"msg": [{"role": "user", "content": in_msg}]}
        else:
            state["msg"].append({"role": "user", "content": in_msg})

        response = bot.invoke(state)
        print(f"Bot: {response['msg'][-1].content}")

