"""
Router functions for the video understanding agent.
"""

from app.router.qa_router import qa_routing
from app.router.reasoning_router import reasoning_routing
from app.router.primitive_question_router import primitive_question_routing

__all__ = ["qa_routing", "reasoning_routing", "primitive_question_routing"]