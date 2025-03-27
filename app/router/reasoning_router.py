"""
Router function for the try_answer_with_reasoning node.
"""

def reasoning_routing(state):
    """
    Routes based on the result of try_answer_with_reasoning.
    
    Args:
        state: The current state
        
    Returns:
        String indicating the next node
    """
    # Implementation will go here
    # For now, placeholder logic
    if 'current_answer' not in state or not state['current_answer']:
        return "not_answered"
    elif state['question_stack'] and len(state['question_stack']) > 1:
        return "answered_not_root"
    else:
        return "answered_root" 