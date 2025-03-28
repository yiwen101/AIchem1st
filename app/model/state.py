"""
State definition for the video understanding agent.
"""

from typing import List, Dict, Any, Optional
from typing_extensions import TypedDict, NotRequired

from app.model.structs import ParquetFileRow, ToolCall, YoutubeVideoInfo, AttemptAnswerResponse, QARecord

# todo, persistence of the state
class VideoAgentState(TypedDict):
    """State for the video understanding agent graph."""
    query: ParquetFileRow
    
    qa_notebook: List[QARecord]
    tool_results: Dict[str, List[object]]

    question_stack: List[str]
    task_queue: List[Dict[str, Any]]

    current_question_tool_results: Dict[str, Any]
    previous_QA: Optional[QARecord]
    prev_attempt_answer_response: Optional[AttemptAnswerResponse] # just to know whether the previous attempt succeeded in answering for conditional routing

def get_current_question(state: VideoAgentState) -> str:
    """Get the current question."""
    return state["question_stack"][-1]

def has_next_question(state: VideoAgentState) -> bool:
    """Check if there is a next question."""
    return len(state["question_stack"]) > 1

def has_pending_tool_calls(state: VideoAgentState) -> bool:
    """Check if there are pending tool calls."""
    return len(state["task_queue"]) > 0

def answer_question(state: VideoAgentState, answer: str, reasoning: str, update_notebook: bool = True) -> None:
    """Answer the current question."""
    current_question = get_current_question(state)
    state["question_stack"].pop()
    qa_pair = QARecord(question=current_question, answer=answer, reason=reasoning)
    state["previous_QA"] = qa_pair
    if update_notebook:
        state["qa_notebook"].append(qa_pair)

def add_new_question(state: VideoAgentState, problem: str) -> None:
    """Add a new question to the question stack."""
    state["question_stack"].append(problem)
    
    state["current_question_tool_results"] = {}
    state["previous_QA"] = None

def add_tool_calls(state: VideoAgentState, tool_calls: List[ToolCall]) -> None:
    """Add tool calls to the task queue."""
    state["task_queue"].extend(tool_calls)

def get_and_clear_pending_tool_calls(state: VideoAgentState) -> List[ToolCall]:
    """Get and clear the pending tool calls."""
    tool_calls = state["task_queue"]
    state["task_queue"] = []
    return tool_calls

def add_tool_result(state: VideoAgentState, tool_name: str, result: object) -> None:
    """Add a tool result to the current question tool results."""
    state["current_question_tool_results"][tool_name] = result
    if tool_name not in state["tool_results"]:
        state["tool_results"][tool_name] = []
    state["tool_results"][tool_name].append(result)


