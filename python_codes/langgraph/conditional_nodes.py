from langgraph.graph import StateGraph, START, END # type: ignore
from typing import List, TypedDict, Literal

class Currency(TypedDict):
    amount_usd: float
    total_usd: float
    total: float
    target_currency: Literal["EUR", "INR", "BRL"]


def total_usd(state: Currency) -> Currency:
    state["total_usd"] = state["amount_usd"] * 1.08
    return state

def target_currency_inr(state: Currency) -> Currency:
    state["total"] = state["total_usd"] * 95.46
    return state

def target_currency_eur(state: Currency) -> Currency:
    state["total"] = state["total_usd"] * 0.91
    return state

def target_currency_brl(state: Currency) -> Currency:
    state["total"] = state["total_usd"] * 3.67
    return state

def choose_conversion(state: Currency) -> str:
    return state["target_currency"]


builder = StateGraph(Currency)

builder.add_node("total_usd",total_usd)
builder.add_node("target_currency_inr",target_currency_inr)
builder.add_node("target_currency_eur",target_currency_eur)
builder.add_node("target_currency_brl",target_currency_brl)

builder.add_edge(START,"total_usd")
builder.add_conditional_edges(
    "total_usd",
    choose_conversion,{
        "INR": "target_currency_inr",
        "EUR": "target_currency_eur",
        "BRL": "target_currency_brl"
    }
)
builder.add_edge("target_currency_inr",END)
builder.add_edge("target_currency_eur",END)
builder.add_edge("target_currency_brl",END)

app = builder.compile()

data= app.invoke({
    "amount_usd": 25,
    "target_currency": "INR"})

print(data)
