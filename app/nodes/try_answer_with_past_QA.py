"""
Node that attempts to answer the current question using past QA records.
"""

from app.common.prompt import format_prompt
from app.common.llm import query_llm_json


node_prompt = ""

node_response_schema = {}
def try_answer_with_past_QA(state):
    """
    Try to answer the current question using the QA notebook.
    
    Args:
        state: The current state dictionary
        
    Returns:
        Updated state with potential answer
    """
    prompt = format_prompt(node_prompt, state, notebook_info=True, output_schema=node_response_schema)
    response = query_llm_json(prompt)
    can_answer = response.get("can_answer", False)
    if can_answer:
        answer = response.get("answer", "")
        reasoning = response.get("reasoning", "")
        state.answer_question(answer, reasoning, update_notebook=False)
        if state.is_root_question():
            return "end"
        else:
            return "parent"
    else:
        return "next"