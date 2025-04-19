"""
Image captioning tools for video analysis.

This module provides tools for generating captions for video frames.
"""

import os
import cv2
import numpy as np
from typing import Dict, Any, List, Optional
import base64
import json
import requests

from app.tools.toolImpl.base_tool import BaseTool, ToolParameter, ToolParameterType
from app.tools.tool_manager import register_tool
from app.common.resource_manager import resource_manager
from app.common.llm.openai import query_vision_llm_single_image

@register_tool
class ImageCaptioningTool(BaseTool):
    """Tool for generating captions for video frames."""
    
    name = "image_captioning"
    description = "Generate a caption for a video frame at specified time."
    parameters = [
        ToolParameter(
            name="time_seconds",
            type=ToolParameterType.FLOAT,
            description="Time in seconds (e.g., 70.45 for 70 seconds and 45 milliseconds)",
            required=True
        )
    ]
    
    @classmethod
    def execute(cls, time_seconds: float) -> Dict[str, Any]:
        """
        Generate a caption for a video frame at specified time.
        
        Args:
            time_seconds: Time in seconds (e.g., 70.45 for 70 seconds and 45 milliseconds)
            
        Returns:
            The generated caption
        """
        # Get frame at specified time
        frame, frame_index = resource_manager.get_frame_at_time(time_seconds)
        
        try:
            # Use OpenAI GPT-4o for image captioning directly with the frame
            caption = query_vision_llm_single_image(frame, "Describe what you see in this image.")
        except Exception as e:
            # Fallback to local captioning if API fails
            caption = cls._local_captioning(frame)
            caption = f"{caption} (API Error: {str(e)})"
        
        # Save captioned image
        output_path = resource_manager.save_captioned_image(
            frame, time_seconds, caption, "image_captioning"
        )

        return {
            "time_seconds": time_seconds,
            "caption": caption,
        }

    @classmethod
    def _local_captioning(cls, image: np.ndarray) -> str:
        """
        Generate a caption locally using a simple heuristic approach.
        
        Args:
            image: Image as numpy array
            
        Returns:
            Generated caption
        """
        # This is a very simple fallback that analyzes basic properties
        height, width, channels = image.shape
        
        # Extract some basic image properties
        avg_color = np.mean(image, axis=(0, 1))
        brightness = np.mean(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        edges = cv2.Canny(image, 100, 200)
        edge_density = np.count_nonzero(edges) / (height * width)
        
        # Very simple heuristic classification
        image_type = "photograph"
        if edge_density > 0.1:
            if brightness > 200:
                image_type = "diagram or drawing"
            else:
                image_type = "detailed image"
        
        # Color description
        color_desc = ""
        if avg_color[0] > 150:
            color_desc = "bluish"
        elif avg_color[1] > 150:
            color_desc = "greenish"
        elif avg_color[2] > 150:
            color_desc = "reddish"
        else:
            if brightness < 50:
                color_desc = "dark"
            elif brightness > 200:
                color_desc = "bright"
            else:
                color_desc = "medium-toned"
        
        # Create caption based on simple analysis
        resolution = "high" if width * height > 1000000 else "medium" if width * height > 400000 else "low"
        orientation = "portrait" if height > width else "landscape" if width > height else "square"
        
        caption = f"A {resolution}-resolution {orientation} {color_desc} {image_type}."
        
        return caption