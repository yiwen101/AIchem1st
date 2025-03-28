
from app.model.state import VideoAgentState
import os

def write_result(state: VideoAgentState):
    """
    Write the result to the result file.
    """
    last_qa = state["previous_QA"]
    if last_qa is None:
        #panic
        raise ValueError("No previous QA found at write_result node")
    
    qid = state["query"].qid
    answer = last_qa["answer"]
    
    #check if the result.csv file exists
    if not os.path.exists("result.csv"):
        with open("result.csv", "w") as f:
            f.write("qid,pred\n")
    
    #append one line to the result.csv file
    with open("result.csv", "a") as f:
        f.write(f"{qid},{answer}\n")
