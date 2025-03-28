"""
Main graph definition for the video understanding agent.
"""

from langgraph.graph import StateGraph, START, END

# Import state definition
from AIchem1st.app.model.state import VideoAgentState

# Import all nodes
from app.nodes.setup import setup
from app.nodes.try_answer_with_past_QA import try_answer_with_past_QA
from app.nodes.try_answer_with_reasoning import try_answer_with_reasoning
from app.nodes.is_primitive_question import is_primitive_question
from app.nodes.decide_tool_calls import decide_tool_calls
from app.nodes.execute_tool_calls import execute_tool_calls
from app.nodes.decompose_to_sub_question import decompose_to_sub_question
from app.nodes.answer_query import answer_query

# Import conditional routing functions from router package
from app.router import (
    qa_routing,
    reasoning_routing,
    primitive_question_routing
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
    graph_builder.add_node("setup", setup)
    graph_builder.add_node("try_answer_with_past_QA", try_answer_with_past_QA)
    graph_builder.add_node("try_answer_with_reasoning", try_answer_with_reasoning)
    graph_builder.add_node("is_primitive_question", is_primitive_question)
    graph_builder.add_node("decide_tool_calls", decide_tool_calls)
    graph_builder.add_node("execute_tool_calls", execute_tool_calls)
    graph_builder.add_node("decompose_to_sub_question", decompose_to_sub_question)
    graph_builder.add_node("answer_query", answer_query)
    
    # Add edges according to the provided flow
    
    # Start to setup
    graph_builder.add_edge(START, "setup")
    
    # Setup to try_answer_with_past_QA
    graph_builder.add_edge("setup", "try_answer_with_past_QA")
    
    # try_answer_with_past_QA conditional routing
    graph_builder.add_conditional_edges(
        "try_answer_with_past_QA",
        qa_routing,
        {
            "not_answered": "try_answer_with_reasoning",
            "answered_not_root": "try_answer_with_past_QA",
            "answered_root": "answer_query"
        }
    )
    
    # try_answer_with_reasoning conditional routing
    graph_builder.add_conditional_edges(
        "try_answer_with_reasoning",
        reasoning_routing,
        {
            "not_answered": "is_primitive_question",
            "answered_not_root": "try_answer_with_past_QA",
            "answered_root": "answer_query"
        }
    )
    
    # is_primitive_question conditional routing
    graph_builder.add_conditional_edges(
        "is_primitive_question",
        primitive_question_routing,
        {
            "yes": "decide_tool_calls",
            "no": "decompose_to_sub_question"
        }
    )
    
    # decide_tool_calls to execute_tool_calls
    graph_builder.add_edge("decide_tool_calls", "execute_tool_calls")
    
    # execute_tool_calls to try_answer_with_reasoning
    graph_builder.add_edge("execute_tool_calls", "try_answer_with_reasoning")
    
    # decompose_to_sub_question to try_answer_with_past_QA
    graph_builder.add_edge("decompose_to_sub_question", "try_answer_with_past_QA")
    
    # answer_query to END
    graph_builder.add_edge("answer_query", END)
    
    # Compile the graph
    return graph_builder.compile()


# Create the graph instance
video_agent_graph = create_video_agent_graph() 