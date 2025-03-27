"""
Basic chatbot implementation using LangGraph with DeepSeek LLM.
"""

from typing import Annotated

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

import os
import sys


    # Get API keys from environment variables
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")

if not deepseek_api_key:
    print("Error: DEEPSEEK_API_KEY environment variable not found.")
    print("Please set it in your .env file or export it to your environment.")
    sys.exit(1)

llm = ChatOpenAI(
    model="deepseek-chat",
    temperature=0,
    openai_api_key=deepseek_api_key,
    openai_api_base='https://api.deepseek.com'
)

class State(TypedDict):
    """State for the chatbot graph."""
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]

def get_user_input(state: State):
    user_input = input("User: ")
    return {"messages": [HumanMessage(content=user_input)]}

def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

def print_messages(state: State):
    print(state["messages"][-1].content)

# if user input is "quit", "exit", "q", then break
def quit_condition(state: State):
    return "end" if state["messages"][-1].content.lower() in ["quit", "exit", "q"] else "chatbot"   

graph_builder = StateGraph(State)
graph_builder.add_node("get_user_input", get_user_input)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("print_messages", print_messages)

graph_builder.add_edge(START, "get_user_input")
graph_builder.add_conditional_edges("get_user_input", quit_condition, {"end": END, "chatbot": "chatbot"})
graph_builder.add_edge("chatbot", "print_messages")
graph_builder.add_edge("print_messages", "get_user_input")

graph = graph_builder.compile()
