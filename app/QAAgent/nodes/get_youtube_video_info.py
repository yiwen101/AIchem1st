from app.model.state import VideoAgentState, add_tool_result
from app.common.utils.youtube import _get_youtube_video_info


# add the video info to the state
def get_youtube_video_info(state: VideoAgentState) -> VideoAgentState:
    tool_name = "youtube_video_info"
    if tool_name in state["current_question_tool_results"]:
        return
    metadata = state["query"]
    youtube_url = metadata.youtube_url
    video_length = metadata.duration
    info = _get_youtube_video_info(youtube_url, video_length)
    add_tool_result(state, tool_name, info)
