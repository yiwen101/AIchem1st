"""
Attempt to answer the current question using the QA notebook.

It simply 
"""

from AIchem1st.app.model.structs import AttemptAnswerResponse
from app.common.prompt import format_prompt
from app.common.llm import query_llm_json
from app.model.state import VideoAgentState
# check if the current question is almost identical to the previous question, if so, answer the question adapting from the previous answer

def get_prompt(question: str) -> str:
    return f"You are a helpful assistant that check whether the new user question is almost identical to any past answered question and attempt to answer the question adapting from the previous answer. The user question is: {question}."

node_response_schema = {
    "can_answer": bool,
    "answer": str,
    "reasoning": str
}


def try_answer_with_past_QA(state: VideoAgentState):
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
    attempt_answer_response = AttemptAnswerResponse(
        can_answer=response["can_answer"],
        answer=response["answer"],
        reasoning=response["reasoning"]
    )
    if attempt_answer_response.can_answer:
        # do not update notebook as a similar question is answered
        state.answer_question(attempt_answer_response.answer, attempt_answer_response.reasoning, update_notebook=False)
    return {"prev_attempt_answer_response": attempt_answer_response}