"""
Max steps routing function for the video agent.

This module provides a routing function that checks if the maximum number of steps 
has been reached and routes to the appropriate next node.
"""

from app.model.state import VideoAgentState, increment_step_count, is_max_steps_reached


def max_steps_routing(state: VideoAgentState, next_steps: dict) -> str:
    """
    Routing function that checks if the maximum number of steps has been reached.
    
    Args:
        state: The current state of the agent
        next_steps: Dictionary of next steps to route to
        
    Returns:
        The next node to route to, or "terminate" if max steps reached
    """
    # Increment step count
    increment_step_count(state)
    
    # Check if max steps reached
    if is_max_steps_reached(state):
        print(f"Maximum steps ({state['max_steps']}) reached for query {state['query'].qid}. Terminating.")
        return "terminate"
    
    # Use the original routing if max steps not reached
    return next_steps 