"""
Main graph definition for the video understanding agent.
"""

from langgraph.graph import StateGraph, START, END

# Import state definition
from app.model.state import VideoAgentState

# Import all nodes
from app.nodes.get_youtube_video_info import get_youtube_video_info
from app.nodes.try_answer_with_past_QA import try_answer_with_past_QA
from app.nodes.try_answer_with_reasoning import try_answer_with_reasoning
from app.nodes.is_primitive_question import is_primitive_question
from app.nodes.execute_tool_calls import execute_tool_calls
from app.nodes.write_result import write_result

# Import conditional routing functions from router package
from app.router.attempt_answer_routing import attempt_answer_routing
from app.router.tool_call_routing import tool_call_routing


def create_video_agent_graph():
    """
    Create and return the video understanding agent graph.
    
    Returns:
        A compiled StateGraph
    """
    # Create state graph with our state definition
    graph_builder = StateGraph(VideoAgentState)
    
    # Add all nodes
    graph_builder.add_node("get_youtube_video_info", get_youtube_video_info)
    graph_builder.add_node("try_answer_with_past_QA", try_answer_with_past_QA)
    graph_builder.add_node("try_answer_with_reasoning", try_answer_with_reasoning)
    graph_builder.add_node("is_primitive_question", is_primitive_question)
    graph_builder.add_node("execute_tool_calls", execute_tool_calls)
    graph_builder.add_node("write_result", write_result)
  
    graph_builder.add_edge(START, "get_youtube_video_info")
    graph_builder.add_edge("get_youtube_video_info", "try_answer_with_past_QA")
    graph_builder.add_conditional_edges(
        "try_answer_with_past_QA",
        attempt_answer_routing,
        {
            "next node": "try_answer_with_reasoning",
            "next question": "try_answer_with_past_QA",
            "end": "write_result"
        }
    )
    graph_builder.add_conditional_edges(
        "try_answer_with_reasoning",
        attempt_answer_routing,
        {
            "next node": "is_primitive_question",
            "next question": "try_answer_with_past_QA",
            "end": "write_result"
        }
    )
    graph_builder.add_conditional_edges(
        "is_primitive_question",
        tool_call_routing,
        {
            "yes": "execute_tool_calls",
            "no": "try_answer_with_past_QA"
        }
    )
    graph_builder.add_edge("execute_tool_calls", "try_answer_with_reasoning")
    graph_builder.add_edge("write_result", END)
    return graph_builder.compile()