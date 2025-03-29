"""
Scene detection tools for video analysis.

This module provides tools for detecting scene changes in videos.
"""

import os
import cv2
import numpy as np
from typing import Dict, Any, List, Tuple, Optional

from app.tools.tool_registry import register_tool

@register_tool("scene_detection")
def scene_detection(
    video_path: str,
    output_dir: str = None,
    threshold: float = 30.0,
    min_scene_length: int = 15
) -> Dict[str, Any]:
    """
    Detect scene changes in a video.
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save scene frames
        threshold: Threshold for scene change detection (higher = less sensitive)
        min_scene_length: Minimum length of a scene in frames
        
    Returns:
        Dictionary with scene boundaries and information
    """
    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create output directory if needed
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Initialize variables
    prev_frame = None
    scene_boundaries = []
    frame_idx = 0
    scene_frames = []
    
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
                    
                    # Save scene change frame if output directory specified
                    if output_dir:
                        frame_path = os.path.join(output_dir, f"scene_{len(scene_boundaries):04d}.jpg")
                        cv2.imwrite(frame_path, frame)
                        scene_frames.append({
                            "scene_idx": len(scene_boundaries),
                            "frame_idx": frame_idx,
                            "time": frame_idx / fps if fps > 0 else 0,
                            "path": frame_path
                        })
        
        # Update previous frame
        prev_frame = gray
        frame_idx += 1
    
    # Release video object
    cap.release()
    
    # Calculate scene information
    scenes = []
    for i, boundary in enumerate(scene_boundaries):
        start_frame = 0 if i == 0 else scene_boundaries[i-1]
        end_frame = boundary
        
        # Add scene details
        scenes.append({
            "scene_idx": i + 1,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "start_time": start_frame / fps if fps > 0 else 0,
            "end_time": end_frame / fps if fps > 0 else 0,
            "duration": (end_frame - start_frame) / fps if fps > 0 else 0
        })
    
    # Add final scene if any scene changes detected
    if scene_boundaries:
        start_frame = scene_boundaries[-1]
        end_frame = frame_count - 1
        
        scenes.append({
            "scene_idx": len(scene_boundaries) + 1,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "start_time": start_frame / fps if fps > 0 else 0,
            "end_time": end_frame / fps if fps > 0 else 0,
            "duration": (end_frame - start_frame) / fps if fps > 0 else 0
        })
    elif frame_count > 0:
        # Only one scene in the entire video
        scenes.append({
            "scene_idx": 1,
            "start_frame": 0,
            "end_frame": frame_count - 1,
            "start_time": 0,
            "end_time": (frame_count - 1) / fps if fps > 0 else 0,
            "duration": frame_count / fps if fps > 0 else 0
        })
    
    return {
        "video_path": video_path,
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "scene_count": len(scenes),
        "scenes": scenes,
        "scene_frames": scene_frames,
        "output_directory": output_dir
    }

@register_tool("extract_frame")
def extract_frame(video_path: str, frame_index: int, output_path: str = None) -> Dict[str, Any]:
    """
    Extract a specific frame from a video.
    
    Args:
        video_path: Path to the video file
        frame_index: Index of the frame to extract
        output_path: Path to save the extracted frame
        
    Returns:
        Dictionary with information about the extracted frame
        
    Raises:
        ValueError: If the video cannot be opened or the frame cannot be read
    """
    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Check if the frame index is valid
    if frame_index < 0 or frame_index >= total_frames:
        raise ValueError(f"Frame index {frame_index} is out of range (0-{total_frames-1})")
    
    # Set the position to the specified frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    
    # Read the frame
    ret, frame = cap.read()
    if not ret:
        raise ValueError(f"Could not read frame {frame_index}")
    
    # Generate output path if not provided
    if not output_path:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = f"frames_{video_name}"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"frame_{frame_index:04d}.jpg")
    
    # Save the frame
    cv2.imwrite(output_path, frame)
    
    # Release the video
    cap.release()
    
    return {
        "frame_index": frame_index,
        "path": output_path,
        "video_path": video_path
    } 