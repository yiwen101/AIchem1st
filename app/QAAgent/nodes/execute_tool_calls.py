from app.model.state import VideoAgentState, get_and_clear_pending_tool_calls, add_tool_result
from app.tools import execute_tool
from app.common.monitor import logger
import os
import sys
import traceback

# todo: take all the pending tool calls from queue, empty the queue; execute them, add the tool call results to both the all tool results and the current_question_tool_results
def execute_tool_calls(state: VideoAgentState):
    """
    Execute the tool calls decided in the previous step.
    
    This function retrieves all pending tool calls from the state's task queue,
    executes each tool with its parameters, and adds the results to the state.
    
    Args:
        state: The current state dictionary
        
    Returns:
        Updated state with tool execution results
    """
    # Get all pending tool calls
    tool_calls = get_and_clear_pending_tool_calls(state)
    
    # Execute each tool call
    for tool_call in tool_calls:
        try:
            tool_name = tool_call.tool_name
            parameters = tool_call.parameters
            
            logger.log_tool_call(tool_name, parameters)
            
            # Fix video_path parameter if it's a placeholder
            if "video_path" in parameters and parameters["video_path"] == "path_to_video":
                # Use the youtube URL from the query
                video_id = state["query"].video_id
                video_path = f"https://www.youtube.com/watch?v={video_id}"
                parameters["video_path"] = video_path
                logger.log_info(f"Updated video_path parameter to: {video_path}")
            
            # Ensure output directory exists if specified
            if "output_dir" in parameters and parameters["output_dir"] and parameters["output_dir"] != "output_directory":
                os.makedirs(parameters["output_dir"], exist_ok=True)
            
            # Execute the tool with parameters
            result = execute_tool(tool_name, **parameters)
            
            # Add result to the state
            add_tool_result(state, tool_name, result)
            logger.log_tool_result(tool_name, "executed successfully")
            
        except Exception as e:
            # More detailed error handling
            error_traceback = traceback.format_exc()
            
            # Handle errors
            error_result = {
                "error": str(e),
                "traceback": error_traceback,
                "parameters": getattr(tool_call, "parameters", "Unknown parameters"),
                "tool_name": getattr(tool_call, "tool_name", "Unknown tool")
            }
            
            # Add error to state
            add_tool_result(state, getattr(tool_call, "tool_name", "unknown_tool"), error_result)
            
            logger.log_error(f"Error executing tool: {error_result['tool_name']}")
            logger.log_error(f"Error message: {error_result['error']}")
            logger.log_error(f"Traceback: {error_traceback}")
    
    return state 