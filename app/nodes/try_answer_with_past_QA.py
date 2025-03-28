"""
Node that attempts to answer the current question using past QA records.
"""

from app.common.prompt import format_prompt
from app.common.llm import query_llm_json

# check if the current question is almost identical to the previous question, if so, answer the question adapting from the previous answer

def get_prompt(question: str) -> str:
    return f"You are a helpful assistant that check whether the new user question is almost identical to any past answered question and attempt to answer the question adapting from the previous answer. The user question is: {question}."

node_response_schema = {
    "can_answer": bool,
    "answer": str,
    "reasoning": str
}


def try_answer_with_past_QA(state):
    """
    Try to answer the current question using the QA notebook.
    
    Args:
        state: The current state dictionary
        
    Returns:
        Updated state with potential answer
    """
    current_question = state.get_current_question()
    prompt = get_prompt(current_question)
    prompt = format_prompt(prompt, state, notebook_info=True, output_schema=node_response_schema)
    response = query_llm_json(prompt)
    return {"prev_attempt_answer_response": response}