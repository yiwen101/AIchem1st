"""
Tools for video analysis and understanding.
"""

# Import base tool classes
from app.tools.toolImpl.base_tool import BaseTool, ToolParameter, ToolParameterType
from app.tools.tool_manager import (
    tool_manager,
    register_tool,
    get_tool_schemas,
    execute_tool,
    format_tools_for_prompt
)

# Import tools
from app.tools.toolImpl.scene_detection import SceneDetectionTool, ExtractFrameTool
from app.tools.toolImpl.object_detection import ObjectDetectionTool
from app.tools.toolImpl.image_captioning import ImageCaptioningTool

# Export primary functions
__all__ = [
    "BaseTool",
    "ToolParameter",
    "ToolParameterType",
    "register_tool",
    "get_tool_schemas",
    "execute_tool",
    "format_tools_for_prompt",
    "SceneDetectionTool",
    "ExtractFrameTool",
    "ObjectDetectionTool",
    "ImageCaptioningTool"
] 