from app.model.state import VideoAgentState, get_and_clear_pending_tool_calls, add_tool_result
from app.tools import execute_tool

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
        tool_name = tool_call.tool_name
        parameters = tool_call.parameters
        
        try:
            # Execute the tool with parameters
            result = execute_tool(tool_name, **parameters)
            
            # Add result to the state
            add_tool_result(state, tool_name, result)
        except Exception as e:
            # Handle errors
            error_result = {
                "error": str(e),
                "parameters": parameters
            }
            add_tool_result(state, tool_name, error_result)
    
    return state 