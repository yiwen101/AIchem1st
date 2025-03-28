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
    prev_attempt_answer_response: Optional[AttemptAnswerResponse] # just to know whether the previous attempt succeeded in answering for conditional routing
    video_info: Optional[YoutubeVideoInfo]
    
    qa_notebook: List[QARecord]
    tool_results: Dict[str, List[object]]

    question_stack: List[str]
    task_queue: List[Dict[str, Any]]

    current_question_tool_results: Dict[str, Any]
    previous_QA: NotRequired[QARecord]
    
    def get_current_question(self) -> str:
        """Get the current question."""
        return self["question_stack"][-1]
    
    def has_next_question(self) -> bool:
        """Check if there is a next question."""
        return len(self["question_stack"]) > 1
    
    def has_pending_tool_calls(self) -> bool:
        """Check if there are pending tool calls."""
        return len(self["task_queue"]) > 0

    def answer_question(self, answer: str, reasoning: str, update_notebook: bool = True) -> None:
        """Answer the current question."""
        current_question = self.current_question()
        self["question_stack"].pop()
        qa_pair = QARecord(question=current_question, answer=answer, reason=reasoning)
        self["previous_QA"] = qa_pair
        if update_notebook:
            self["qa_notebook"].append(qa_pair)
    
    def add_new_question(self, problem: str) -> None:
        """Add a new question to the question stack."""
        self["question_stack"].append(problem)
        
        self["current_question_tool_results"] = {}
        self["previous_QA"] = None
    
    def add_tool_calls(self, tool_calls: List[ToolCall]) -> None:
        """Add tool calls to the task queue."""
        self["task_queue"].extend(tool_calls)
    
    def get_and_clear_pending_tool_calls(self) -> List[ToolCall]:
        """Get and clear the pending tool calls."""
        tool_calls = self["task_queue"]
        self["task_queue"] = []
        return tool_calls
    
    def add_tool_result(self, tool_name: str, result: object) -> None:
        """Add a tool result to the current question tool results."""
        self["current_question_tool_results"][tool_name] = result
        self["tool_results"][tool_name].append(result)


