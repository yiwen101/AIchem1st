"""
State definition for the video understanding agent.
"""

from typing import List, Dict, Any, Optional
from typing_extensions import TypedDict, NotRequired

from app.model.structs import ParquetFileRow


class QARecord(TypedDict):
    """A record of a question, its answer, and the reasoning behind it."""
    question: str
    answer: Optional[str]
    reason: Optional[str]


class VideoAgentState(TypedDict):
    """State for the video understanding agent graph."""
    query: ParquetFileRow
    video_filename: str
    qa_notebook: List[QARecord]
    tool_results: Dict[str, List[str]]

    question_stack: List[str]
    task_queue: List[Dict[str, Any]]

    current_question_tool_results: Dict[str, Any]
    previous_QA: NotRequired[QARecord]

    def __init__(self, filename: str, question: str):
        self["video_filename"] = filename
        self["qa_notebook"] = []
        self["tool_results"] = {}
        
        self["question_stack"] = [question]
        self["task_queue"] = []

        self["current_question_tool_results"] = {}
        self["previous_QA"] = None

    def is_root_question(self) -> bool:
        """Check if the current question is the root question."""
        return len(self["question_stack"]) == 1
    
    def current_question(self) -> str:
        """Get the current question."""
        return self["question_stack"][-1]

    def answer_question(self, answer: str, reasoning: str, update_notebook: bool = True) -> None:
        """Answer the current question."""
        current_question = self.current_question()
        self["question_stack"].pop()
        qa_pair = QARecord(question=current_question, answer=answer, reason=reasoning)
        self["previous_QA"] = qa_pair
        if update_notebook:
            self["qa_notebook"].append(qa_pair)

