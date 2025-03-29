"""
Tool registry for video analysis agent.

This module provides a central registry of all available tools that can be used
by the video analysis agent.
"""

from typing import Dict, Callable, Any

# Dictionary to store all registered tools
_TOOLS: Dict[str, Callable] = {}

def register_tool(name: str):
    """
    Decorator to register a function as a tool.
    
    Args:
        name: The name of the tool
    
    Returns:
        Decorator function that registers the tool
    """
    def decorator(func: Callable):
        _TOOLS[name] = func
        return func
    return decorator

def get_available_tools() -> Dict[str, Callable]:
    """
    Get all registered tools.
    
    Returns:
        Dictionary of tool names to tool functions
    """
    return _TOOLS

def execute_tool(tool_name: str, **kwargs) -> Any:
    """
    Execute a tool by name with the given parameters.
    
    Args:
        tool_name: Name of the tool to execute
        **kwargs: Parameters to pass to the tool
    
    Returns:
        The result of the tool execution
    
    Raises:
        ValueError: If the tool is not found
    """
    if tool_name not in _TOOLS:
        raise ValueError(f"Tool '{tool_name}' not found. Available tools: {list(_TOOLS.keys())}")
    
    return _TOOLS[tool_name](**kwargs) 