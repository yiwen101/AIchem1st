"""
Tools for video analysis and understanding.
"""

# Import base tool classes
from app.tools.base_tool import BaseTool, ToolParameter, ToolParameterType
from app.tools.tool_manager import (
    tool_manager,
    register_cv_tool,
    get_tool_schemas,
    execute_tool,
    format_tools_for_prompt
)

# Import tools
from app.tools.scene_detection import scene_detection
from app.tools.object_detection import object_detection
from app.tools.video_frames import extract_frames, separate_background_foreground
from app.tools.image_captioning import image_captioning

# Import CV tools package
from app.tools import cv

# Register legacy tools
tool_manager.register_legacy_tools()

# Export primary functions
__all__ = [
    "BaseTool",
    "ToolParameter",
    "ToolParameterType",
    "register_cv_tool",
    "get_tool_schemas",
    "execute_tool",
    "format_tools_for_prompt",
    "scene_detection",
    "object_detection",
    "extract_frames",
    "separate_background_foreground",
    "image_captioning"
] 