from app.model.state import VideoAgentState, has_next_question

'''
Check whether the previous attempt answer response node did answer, and route accordingly

if the question is not answered, go to next node
else, if the question answered has no parent question (namely it is the root question), can end the execution
else, recursively go to answer the parent question
'''

def attempt_answer_routing(state: VideoAgentState):
    prev_attempt_answer_response = state.get('prev_attempt_answer_response', None)
    if prev_attempt_answer_response is None:
        return "next node"
    
    if not prev_attempt_answer_response.can_answer:
        return "next node"
        
    if has_next_question(state):
        return "next question"
    else:
        return "end"