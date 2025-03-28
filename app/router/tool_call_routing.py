
def tool_call_routing(state):
    """
    Routes based on the tool call result.
    """
    return "yes" if state.has_pending_tool_calls() else "no"

