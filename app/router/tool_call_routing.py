from app.model.state import VideoAgentState

def tool_call_routing(state: VideoAgentState):
    return "yes" if state.has_pending_tool_calls() else "no"

