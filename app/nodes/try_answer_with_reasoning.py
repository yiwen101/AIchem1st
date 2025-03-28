"""
Node that attempts to answer the current question using reasoning.
"""

from app.common.prompt import format_prompt
from app.common.llm import query_llm_json

def get_prompt(question: str) -> str:
    return f"You are a helpful assistant that check whether the new user question can be answered by deducing from existing information. The user question is: {question}."

node_response_schema = {
    "can_answer": bool,
    "answer": str,
    "reasoning": str
}

def try_answer_with_reasoning(state):
    """
    Try to answer the current question using reasoning capabilities.
    
    Args:
        state: The current state dictionary
        
    Returns:
        Updated state with potential answer based on reasoning
    """
    current_question = state.get_current_question()
    prompt = get_prompt(current_question)
    prompt = format_prompt(prompt, state, notebook_info=True, tool_call_info=True, pre_question_info=True, current_question_tool_results=True, output_schema=node_response_schema)
    response = query_llm_json(prompt)
    return {"prev_attempt_answer_response": response}
