"""
Router function for the is_primitive_question node.
"""

def primitive_question_routing(state):
    """
    Routes based on whether the question is primitive or complex.
    
    Args:
        state: The current state
        
    Returns:
        String indicating the next node
    """
    # Implementation will go here
    # For now, placeholder logic
    if state.get('is_primitive', False):
        return "yes"
    else:
        return "no" 