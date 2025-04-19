"""
Node that attempts to answer the current question using reasoning.
"""

from app.QAAgent.prompt import generate_prompt
from app.common.llm import query_llm_json
from app.common.monitor import logger
from app.model.structs import AttemptAnswerResponse
from app.model.state import VideoAgentState, answer_question, get_current_question
import json

def get_prompt(question: str) -> str:
    return f"You are a helpful assistant that check whether the new user question can be answered by deducing from existing information. The user question is: {question}."

# Define schema as a proper JSON-serializable dictionary
node_response_schema = {
    "type": "object",
    "properties": {
        "can_answer": {
            "type": "boolean",
            "description": "Whether the question can be answered with current information"
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

def try_answer_with_reasoning(state: VideoAgentState):
    """
    Try to answer the current question using reasoning capabilities.
    
    Args:
        state: The current state dictionary
        
    Returns:
        Updated state with potential answer based on reasoning
    """
    current_question = get_current_question(state)
    prompt = get_prompt(current_question)
    prompt = generate_prompt(
        prompt, 
        state, 
        notebook_info=True, 
        tool_call_info=True, 
        pre_question_info=True, 
        current_question_tool_results=True, 
        output_schema=json.dumps(node_response_schema)
    )
    response = query_llm_json(prompt, reasoning=True)
    
    # Create attempt answer response object
    attempt_answer_response = AttemptAnswerResponse(
        can_answer=response["can_answer"],
        answer=response["answer"],
        reasoning=response["reasoning"]
    )
    
    # Store in state for routing
    state["prev_attempt_answer_response"] = attempt_answer_response
    
    # If we can answer, also call answer_question to update the previous_QA field
    if attempt_answer_response.can_answer:
        answer_question(state, attempt_answer_response.answer, attempt_answer_response.reasoning, update_notebook=True)
    
    return state
