"""
Determine if the current question is primitive or complex.

primitive question can be handled without considering a sub problem after doing consective tool calls.
complex question should be decomposed into a sub problem first, then answer the sub problem

this code should either append the state with a sub problem, or append the state with tool calls
"""

from typing import List

from app.common.prompt import format_prompt
from app.common.llm import query_llm_json
from app.model.structs import ToolCall

def get_prompt(question: str) -> str:
    return f"You are a helpful assistant that check whether it is more suitable to answer the user query by considering a sub problem, or getting direct observations by making tool call(s). The user query is: {question}. If you feel it is more suitable to answer the question by considering a sub problem, please return the sub problem. Otherwise, return the array of tool call(s) (with parameters filled) of length no more than 3 that can be used to answer the question"

node_response_schema = {
    "sub_problem": str|None,
    "tool_calls": List[ToolCall]|None
}


def is_primitive_question(state):
    current_question = state.get_current_question()
    prompt = get_prompt(current_question)
    prompt = format_prompt(prompt, state, notebook_info=True, tool_call_info=True, add_tool_info=True, output_schema=node_response_schema)
    response = query_llm_json(prompt)
    has_sub_problem = response["sub_problem"] is not None
    has_tool_calls = response["tool_calls"] is not None
    if has_sub_problem:
        state.add_problem(response["sub_problem"])
    if has_tool_calls:
        state.add_tool_calls(response["tool_calls"])
