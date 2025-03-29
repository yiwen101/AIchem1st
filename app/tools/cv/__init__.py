"""
Computer Vision (CV) tools for the video agent system.

This package provides tools for video and image analysis, including:
- Video frame extraction
- Background/foreground separation
- Image captioning
- Object detection
- Scene detection
- Video summarization
"""

# Import all CV tools to ensure they are registered
from app.tools.cv import video_frames
from app.tools.cv import image_captioning
from app.tools.cv import video_summarization

# Export the modules
__all__ = [
    "video_frames",
    "image_captioning", 
    "video_summarization"
] 