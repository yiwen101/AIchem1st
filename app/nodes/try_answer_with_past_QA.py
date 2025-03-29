"""
Attempt to answer the current question using the QA notebook.

It simply 
"""

from app.model.structs import AttemptAnswerResponse
from app.common.prompt import generate_prompt
from app.common.llm import query_llm_json
from app.model.state import VideoAgentState, answer_question, get_current_question
import json

# check if the current question is almost identical to the previous question, if so, answer the question adapting from the previous answer

def get_prompt(question: str) -> str:
    return f"You are a helpful assistant that check whether the new user question is almost identical to any past answered question and attempt to answer the question adapting from the previous answer. The user question is: {question}."

# Define schema as a proper JSON-serializable dictionary
node_response_schema = {
    "type": "object",
    "properties": {
        "can_answer": {
            "type": "boolean",
            "description": "Whether the question can be answered based on past Q&A"
        },
        "answer": {
            "type": "string",
            "description": "The answer to the question if can_answer is true"
        },
        "reasoning": {
            "type": "string",
            "description": "The reasoning process behind the answer or why it can't be answered"
        }
    },
    "required": ["can_answer", "answer", "reasoning"]
}


def try_answer_with_past_QA(state: VideoAgentState):
    """
    Try to answer the current question using the QA notebook.
    
    Args:
        state: The current state dictionary
        
    Returns:
        Updated state with potential answer
    """
    current_question = get_current_question(state)
    prompt = get_prompt(current_question)
    prompt = generate_prompt(
        prompt, 
        state, 
        notebook_info=True, 
        output_schema=json.dumps(node_response_schema)
    )
    response = query_llm_json(prompt)
    print(f"LLM response: {response}")
    
    # Create attempt answer response object
    attempt_answer_response = AttemptAnswerResponse(
        can_answer=response["can_answer"],
        answer=response["answer"],
        reasoning=response["reasoning"]
    )
    
    # Store in state for routing
    state["prev_attempt_answer_response"] = attempt_answer_response
    
    if attempt_answer_response.can_answer:
        # do not update notebook as a similar question is answered
        answer_question(state, attempt_answer_response.answer, attempt_answer_response.reasoning, update_notebook=False)
        print(f"Successfully answered question using past QA: {current_question}")
        print(f"Answer: {attempt_answer_response.answer}")
    else:
        print(f"Could not answer question using past QA: {current_question}")
    
    return state