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
from app.tools.toolImpl.scene_detection import SceneDetectionTool
from app.tools.toolImpl.object_detection import ObjectDetectionTool
from app.tools.toolImpl.image_captioning import ImageCaptioningTool
from app.tools.toolImpl.object_tracking import ObjectTrackingTool
from app.tools.toolImpl.optical_flow_based_object_tracking import OpticalFlowTrackingTool
from app.tools.toolImpl.background_based_object_tracking import BackgroundBasedTrackingTool
from app.tools.toolImpl.llm_based_motion_detection import LLMBasedMotionDetectionTool

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
    "ObjectDetectionTool",
    "ImageCaptioningTool",
    "ObjectTrackingTool",
    "OpticalFlowTrackingTool",
    "BackgroundBasedTrackingTool",
    "LLMBasedMotionDetectionTool"
] 