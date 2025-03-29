from app.model.state import VideoAgentState, add_tool_result
from app.tools.tool_manager import execute_tool
from app.common.monitor import logger

def get_scene_info(state: VideoAgentState) -> VideoAgentState:
    """
    Execute the scene detection tool and add the results to the state.
    
    This node is executed after getting YouTube video info and 
    automatically runs the scene detection tool with default parameters.
    
    Args:
        state: The current state dictionary
        
    Returns:
        Updated state with scene detection results
    """
    tool_name = "scene_detection"
    
    # Skip if we already have scene detection results for this question
    if tool_name in state["current_question_tool_results"]:
        logger.log_info(f"Skipping {tool_name} as it was already executed")
        return state
    
    try:
        logger.log_info(f"Executing {tool_name} with default parameters")
        
        # Execute the scene detection tool with default parameters
        result = execute_tool(tool_name)
        
        # Add result to the state
        add_tool_result(state, tool_name, result)
        logger.log_info(f"Successfully added {tool_name} results to state")
        
    except Exception as e:
        # Handle errors
        error_message = f"Error executing {tool_name}: {str(e)}"
        logger.log_error(error_message)
        
        # Add error to state as the tool result
        add_tool_result(state, tool_name, {"error": error_message})
    
    return state 