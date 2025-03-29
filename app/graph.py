"""
Main graph definition for the video understanding agent.
"""

from langgraph.graph import StateGraph, START, END

# Import state definition
from app.model.state import VideoAgentState, increment_step_count, is_max_steps_reached

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


# Define a step checking function that wraps other routers
def with_step_check(original_router):
    """
    Wrap a router function with step counting check.
    
    Args:
        original_router: The original router function
        
    Returns:
        A new router function that checks max steps first
    """
    def wrapped_router(state):
        # First increment the step count
        increment_step_count(state)
        
        # Check if we've reached max steps
        if is_max_steps_reached(state):
            print(f"Maximum steps ({state['max_steps']}) reached for query {state['query'].qid}. Terminating.")
            return "write_result"
        
        # Otherwise use the original router
        return original_router(state)
    
    return wrapped_router


# Custom router for execute_tool_calls
def execute_tool_calls_router(state):
    """Router after execute_tool_calls."""
    # Always go back to try_answer_with_reasoning
    return "try_answer_with_reasoning"


def create_video_agent_graph(max_steps: int = 10):
    """
    Create and return the video understanding agent graph.
    
    Args:
        max_steps: Maximum number of steps before terminating
    
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
    
    # Set up the basic flow
    graph_builder.add_edge(START, "get_youtube_video_info")
    graph_builder.add_edge("get_youtube_video_info", "try_answer_with_past_QA")
    
    # Connect try_answer_with_past_QA with step checking
    graph_builder.add_conditional_edges(
        "try_answer_with_past_QA",
        with_step_check(attempt_answer_routing),
        {
            "next node": "try_answer_with_reasoning",
            "next question": "try_answer_with_past_QA",
            "write_result": "write_result",
            "end": "write_result"
        }
    )
    
    # Connect try_answer_with_reasoning with step checking
    graph_builder.add_conditional_edges(
        "try_answer_with_reasoning",
        with_step_check(attempt_answer_routing),
        {
            "next node": "is_primitive_question",
            "next question": "try_answer_with_past_QA",
            "write_result": "write_result",
            "end": "write_result"
        }
    )
    
    # Connect is_primitive_question with step checking
    graph_builder.add_conditional_edges(
        "is_primitive_question",
        with_step_check(tool_call_routing),
        {
            "yes": "execute_tool_calls",
            "no": "try_answer_with_past_QA",
            "write_result": "write_result"
        }
    )
    
    # Connect execute_tool_calls with step checking
    graph_builder.add_conditional_edges(
        "execute_tool_calls",
        with_step_check(execute_tool_calls_router),
        {
            "try_answer_with_reasoning": "try_answer_with_reasoning",
            "write_result": "write_result"
        }
    )
    
    # Connect write_result to the END node
    graph_builder.add_edge("write_result", END)
    
    return graph_builder.compile()