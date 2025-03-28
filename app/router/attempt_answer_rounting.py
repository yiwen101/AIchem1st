def attempt_answer_routing(state):
    """
    Routes based on whether the question is primitive or complex.
    
    Args:
        state: The current state
        
    Returns:
        String indicating the next node
    """
    prev_attempt_answer_response = state.get('prev_attempt_answer_response', None)
    if prev_attempt_answer_response is None:
        return "next node"
    
    if not prev_attempt_answer_response.can_answer:
        return "next node"
   
    if state.has_next_question():
        return "next question"
    else:
        return "end"