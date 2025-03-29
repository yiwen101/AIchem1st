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
from app.tools.resource.resource_manager import resource_manager

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
    def execute(cls, threshold: float = 30.0, min_scene_length: int = 15) -> List[Dict[str, Any]]:
        """
        Detect scene changes in the active video.
        
        Args:
            threshold: Threshold for scene change detection (higher = less sensitive), default is 30.0
            min_scene_length: Minimum length of a scene in frames, default is 15
            
        Returns:
            List of objects with format {"start_time": float, "end_time": float, "duration": float} that describe the scene changes in the video
        """
        # Get active video
        cap, metadata = resource_manager.get_active_video()
        
        # Get video properties
        fps = metadata["fps"]
        frame_count = metadata["frame_count"]
        width = metadata["width"]
        height = metadata["height"]
        video_name = metadata["video_name"]
        
        # Create output directory
        output_dir = f"app/tools/output/scene_detection/{video_name}/{threshold}_{min_scene_length}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize variables
        prev_frame = None
        scene_boundaries = []
        frame_idx = 0
        
        # Reset video position
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Process each frame
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert to grayscale for comparison
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Calculate difference from previous frame
            if prev_frame is not None:
                # Mean absolute difference between frames
                diff = cv2.absdiff(gray, prev_frame)
                diff_mean = np.mean(diff)
                
                # Detect scene change if difference exceeds threshold
                if diff_mean > threshold:
                    # Ensure minimum scene length
                    if not scene_boundaries or (frame_idx - scene_boundaries[-1]) >= min_scene_length:
                        scene_boundaries.append(frame_idx)
                        
                        # Save scene change frame
                        frame_path = os.path.join(output_dir, f"scene_{len(scene_boundaries):04d}.jpg")
                        cv2.imwrite(frame_path, frame)
            
            # Update previous frame
            prev_frame = gray
            frame_idx += 1
        
        # Calculate scene information
        scenes = []
        for i, boundary in enumerate(scene_boundaries):
            start_frame = 0 if i == 0 else scene_boundaries[i-1]
            end_frame = boundary
            
            # Add scene details with simplified output
            scenes.append({
                "start_time": start_frame / fps if fps > 0 else 0,
                "end_time": end_frame / fps if fps > 0 else 0,
                "duration": (end_frame - start_frame) / fps if fps > 0 else 0
            })
        
        # Add final scene if any scene changes detected
        if scene_boundaries:
            start_frame = scene_boundaries[-1]
            end_frame = frame_count - 1
            
            scenes.append({
                "start_time": start_frame / fps if fps > 0 else 0,
                "end_time": end_frame / fps if fps > 0 else 0,
                "duration": (end_frame - start_frame) / fps if fps > 0 else 0
            })
        elif frame_count > 0:
            # Only one scene in the entire video
            scenes.append({
                "start_time": 0,
                "end_time": (frame_count - 1) / fps if fps > 0 else 0,
                "duration": frame_count / fps if fps > 0 else 0
            })
        
        return scenes


@register_tool
class ExtractFrameTool(BaseTool):
    """Tool for extracting a specific frame from a video."""
    
    name = "extract_frame"
    description = "Extract a specific frame from the active video at the given time."
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
        Extract a specific frame from the active video at the given time.
        
        Args:
            time_seconds: Time in seconds (e.g., 70.45 for 70 seconds and 45 milliseconds)
            
        Returns:
            Dictionary with information about the extracted frame
        """
        # Get frame at specified time
        frame, frame_index = resource_manager.get_frame_at_time(time_seconds)
        
        # Get active video metadata
        _, metadata = resource_manager.get_active_video()
        video_name = metadata["video_name"]
        
        # Generate output path
        output_dir = f"app/tools/output/frame_extraction/{video_name}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Format time as SS_MS
        time_formatted = f"{int(time_seconds):02d}_{int((time_seconds % 1) * 1000):03d}"
        output_path = os.path.join(output_dir, f"frame_{time_formatted}.jpg")
        
        # Save the frame
        cv2.imwrite(output_path, frame)
        
        return {
            "frame_index": frame_index,
            "path": output_path,
            "time": time_seconds
        } 