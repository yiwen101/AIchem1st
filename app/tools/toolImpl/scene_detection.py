"""
Scene detection tools for video analysis.

This module provides tools for detecting scene changes in videos.
"""

import os
import cv2
import numpy as np
from typing import Dict, Any, List, Tuple, Optional

from app.tools.toolImpl.base_tool import BaseTool, ToolParameter, ToolParameterType
from app.tools.tool_manager import register_tool
from app.common.resource_manager.resource_manager import resource_manager, ResourceManager

from app.common.utils.scene import detect_scenes
@register_tool
class SceneDetectionTool(BaseTool):
    """Tool for detecting scene changes in a video."""
    
    name = "scene_detection"
    description = "Detect scene changes in the active video."
    parameters = [
        ToolParameter(
            name="threshold",
            type=ToolParameterType.FLOAT,
            description="Threshold for scene change detection (higher = less sensitive)",
            required=False,
            default=30.0
        ),
        ToolParameter(
            name="min_scene_length",
            type=ToolParameterType.INTEGER,
            description="Minimum length of a scene in frames",
            required=False,
            default=15
        )
    ]
    
    @classmethod
    def execute(cls, threshold: float = 30.0, min_scene_length: int = 15) -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
        imgs, infos = detect_scenes(resource_manager, threshold, min_scene_length)
        return infos