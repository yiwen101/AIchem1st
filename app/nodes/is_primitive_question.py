"""
Determine if the current question is primitive or complex.

primitive question can be handled without considering a sub problem after doing consective tool calls.
complex question should be decomposed into a sub problem first, then answer the sub problem

this code should either append the state with a sub problem, or append the state with tool calls
"""

from typing import List
import json

from app.common.prompt import generate_prompt
from app.common.llm import query_llm_json
from app.model.structs import ToolCall
from app.model.state import VideoAgentState, add_tool_calls, add_new_question, get_current_question

def get_prompt(question: str) -> str:
    return f"You are a helpful assistant that check whether it is more suitable to answer the user query by considering a sub problem, or getting direct observations by making tool call(s). The user query is: {question}. If you feel it is more suitable to answer the question by considering a sub problem, please return the sub problem. Otherwise, return the array of tool call(s) (with parameters filled) of length no more than 3 that can be used to answer the question"

node_response_schema = {
    "sub_problem": str|None,
    "tool_calls": List[ToolCall]|None
}


def is_primitive_question(state: VideoAgentState):
    current_question = get_current_question(state)
    prompt = get_prompt(current_question)
    prompt = generate_prompt(
        prompt, 
        state,
        notebook_info=True, 
        tool_call_info=True, 
        add_tool_info=True, 
        output_schema=json.dumps(node_response_schema)
    )
    response = query_llm_json(prompt)
    has_sub_problem = response.get("sub_problem") is not None
    has_tool_calls = response.get("tool_calls") is not None
    if has_sub_problem:
        add_new_question(state, response["sub_problem"])
    if has_tool_calls:
        add_tool_calls(state, response["tool_calls"])
    
    return state
