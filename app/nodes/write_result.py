from app.model.state import VideoAgentState, is_max_steps_reached, get_current_question
import os

def write_result(state: VideoAgentState):
    """
    Write the result to the result file.
    """
    last_qa = state["previous_QA"]
    qid = state["query"].qid
    
    # Check if we hit max steps without an answer
    if last_qa is None:
        if is_max_steps_reached(state):
            print(f"Warning: Maximum steps reached for {qid} without completing analysis.")
            # Create a default answer indicating we hit the step limit
            current_question = get_current_question(state) if state["question_stack"] else state["query"].question
            answer = "INCOMPLETE due to step limit"
        else:
            # This means we've reached write_result but no answer was generated
            # This can happen if routing was incorrect or there were errors in previous nodes
            print(f"Warning: No answer generated for {qid}, but max steps not reached. Possible routing issue.")
            answer = "ERROR: No answer generated"
    else:
        # Access the answer field from the QARecord object
        answer = last_qa.answer
    
    # Check if the result.csv file exists
    if not os.path.exists("result.csv"):
        with open("result.csv", "w") as f:
            f.write("qid,pred\n")
    
    # Append one line to the result.csv file
    with open("result.csv", "a") as f:
        f.write(f"{qid},{answer}\n")
    
    return state
