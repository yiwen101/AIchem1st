"""
Main graph definition for the video understanding agent.
"""

from langgraph.graph import StateGraph, START, END

# Import state definition
from AIchem1st.app.model.state import VideoAgentState

# Import all nodes
from app.nodes.get_youtube_video_info import get_youtube_video_info
from app.nodes.try_answer_with_past_QA import try_answer_with_past_QA
from app.nodes.try_answer_with_reasoning import try_answer_with_reasoning
from app.nodes.is_primitive_question import is_primitive_question
from app.nodes.execute_tool_calls import execute_tool_calls
from app.nodes.decompose_to_sub_question import decompose_to_sub_question
from app.nodes.write_result import write_result

# Import conditional routing functions from router package
from app.router import (
    attempt_answer_routing,
    tool_call_routing
)


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
    graph_builder.add_node("decompose_to_sub_question", decompose_to_sub_question)
    graph_builder.add_node("write_result", write_result)
  
    
    # Add edges according to the provided flow
    
    # Start to setup
    graph_builder.add_edge(START, "get_youtube_video_info")
    
    # get_youtube_video_info to try_answer_with_past_QA
    graph_builder.add_edge("get_youtube_video_info", "try_answer_with_past_QA")
    
    # try_answer_with_past_QA conditional routing
    graph_builder.add_conditional_edges(
        "try_answer_with_past_QA",
        attempt_answer_routing,
        {
            "next node": "try_answer_with_reasoning",
            "next question": "try_answer_with_past_QA",
            "end": "write_result"
        }
    )
    
    # try_answer_with_reasoning conditional routing
    graph_builder.add_conditional_edges(
        "try_answer_with_reasoning",
        attempt_answer_routing,
        {
            "next node": "is_primitive_question",
            "next question": "try_answer_with_past_QA",
            "end": "write_result"
        }
    )
    
    # is_primitive_question conditional routing
    graph_builder.add_conditional_edges(
        "is_primitive_question",
        tool_call_routing,
        {
            "yes": "execute_tool_calls",
            "no": "decompose_to_sub_question"
        }
    )
    
    # execute_tool_calls to try_answer_with_reasoning
    graph_builder.add_edge("execute_tool_calls", "try_answer_with_reasoning")
    
    # decompose_to_sub_question to try_answer_with_past_QA
    graph_builder.add_edge("decompose_to_sub_question", "try_answer_with_past_QA")
    
    # answer_query to END
    graph_builder.add_edge("write_result", END)
    
    # Compile the graph
    return graph_builder.compile()


# Create the graph instance
video_agent_graph = create_video_agent_graph() 