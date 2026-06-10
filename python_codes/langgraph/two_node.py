from langgraph.graph import StateGraph, START, END # type: ignore
from typing import List, TypedDict

class Ecommerce(TypedDict): 
    items: List[float] = []
    subtotal: float
    grand_total: float

def subtotal_node(state: Ecommerce) -> Ecommerce:
    state["subtotal"] = sum(state["items"])
    print(f"Subtotal: {state['subtotal']}")
    return state

def grand_total_node(state: Ecommerce) -> Ecommerce:
    state["grand_total"] = state["subtotal"] * 1.10
    print(f"Grand Total: {state['grand_total']}")
    return state


builder = StateGraph(Ecommerce)

builder.add_node("subtotal",subtotal_node)
builder.add_node("grand_total",grand_total_node)


builder.add_edge(START,"subtotal")
builder.add_edge("subtotal","grand_total")
builder.add_edge("grand_total",END)

graph = builder.compile()

graph.invoke({
    "items": [10, 20, 30],
    "subtotal": 0,
    "grand_total": 0})  