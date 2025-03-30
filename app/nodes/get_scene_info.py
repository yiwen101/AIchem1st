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
    result = execute_tool(tool_name)
    add_tool_result(state, tool_name, result)
    
    important_times = [0]
    for scene in result:
        end_time = scene["end_time"]
        important_times.append(end_time)

        prev_time = important_times[-1]
        #mid_time = (end_time + prev_time) / 2
        #important_times.append(mid_time)
        

    tool_name = "image_captioning"
    for time in important_times:
        tool_result = execute_tool(tool_name, time_seconds=time)
        add_tool_result(state, tool_name, tool_result)
    logger.log_info(f"Successfully added {tool_name} results to state")
    
    '''
    tool_name = "object_detection"
    for time in important_times:
        tool_result = execute_tool(tool_name, time_seconds=time)
        add_tool_result(state, tool_name, tool_result)

    logger.log_info(f"Successfully added {tool_name} results to state")
    '''

    return {
        "current_question_tool_results": {}
    }
    