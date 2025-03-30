from app.model.state import VideoAgentState, has_pending_tool_calls

def tool_call_routing(state: VideoAgentState):
    return "yes" if has_pending_tool_calls(state) else "no"

