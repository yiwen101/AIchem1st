"""
State definition for the video understanding agent.
"""

from typing import List, Dict, Any, Optional
from typing_extensions import TypedDict, NotRequired


class QARecord(TypedDict):
    """A record of a question, its answer, and the reasoning behind it."""
    question: str
    answer: Optional[str]
    reason: Optional[str]


class VideoAgentState(TypedDict):
    """State for the video understanding agent graph."""
    video_filename: str
    qa_notebook: List[QARecord]
    tool_results: Dict[str, List[str]]

    question_stack: List[str]
    task_queue: List[Dict[str, Any]]

    current_question_tool_results: Dict[str, Any]
    previous_QA: NotRequired[QARecord]