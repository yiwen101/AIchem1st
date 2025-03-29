"""
Video frame extraction and processing tools.

This module provides functionality to extract frames from videos at regular
intervals and to separate background/foreground.
"""

import os
import cv2
import numpy as np
from typing import Dict, Any, List

from app.tools.tool_registry import register_tool

@register_tool("extract_frames")
def extract_frames(
    video_path: str,
    output_dir: str = None,
    interval: float = 1.0,
    max_frames: int = 100
) -> Dict[str, Any]:
    """
    Extract frames from a video at regular intervals.
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save frames (if None, generated based on video name)
        interval: Interval in seconds between frames
        max_frames: Maximum number of frames to extract
        
    Returns:
        Dictionary with frame paths and video information
        
    Raises:
        ValueError: If the video cannot be opened
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
    duration = frame_count / fps if fps > 0 else 0
    
    # Calculate frames to save based on interval
    frames_interval = int(fps * interval)
    if frames_interval < 1:
        frames_interval = 1
    
    # Create output directory if it doesn't exist
    if not output_dir:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = f"{video_name}_frames"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract frames at specified intervals
    saved_frames = []
    frame_count_saved = 0
    
    for frame_position in range(0, frame_count, frames_interval):
        # Limit number of frames
        if frame_count_saved >= max_frames:
            break
            
        # Set the position to the exact frame we want
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_position)
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Save the frame
        frame_path = os.path.join(output_dir, f"frame_{frame_count_saved:04d}.jpg")
        cv2.imwrite(frame_path, frame)
        
        # Record frame information
        time_position = frame_position / fps if fps > 0 else 0
        saved_frames.append({
            "frame_index": frame_position,
            "time_position": time_position,
            "path": frame_path
        })
        
        frame_count_saved += 1
    
    # Release video object
    cap.release()
    
    return {
        "video_path": video_path,
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration": duration,
        "frames_saved": frame_count_saved,
        "frames": saved_frames,
        "output_directory": output_dir
    }

@register_tool("separate_background_foreground")
def separate_background_foreground(
    video_path: str,
    output_dir: str = None,
    sample_frames: int = 500,
    save_frames: bool = True,
    interval: float = 1.0
) -> Dict[str, Any]:
    """
    Separate background and foreground from a video.
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save outputs (if None, generated based on video name)
        sample_frames: Number of frames to use for learning background
        save_frames: Whether to save sample frames from background/foreground
        interval: Interval in seconds between saved frames
        
    Returns:
        Dictionary with paths to background/foreground videos and frames
        
    Raises:
        ValueError: If the video cannot be opened
    """
    # Create output directory if it doesn't exist
    if not output_dir:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = f"{video_name}_bg_fg"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create background subtractor
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=sample_frames, 
        varThreshold=16, 
        detectShadows=True
    )
    
    # Create output video writers
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    bg_video_path = os.path.join(output_dir, 'background.mp4')
    fg_video_path = os.path.join(output_dir, 'foreground.mp4')
    
    bg_writer = cv2.VideoWriter(bg_video_path, fourcc, fps, (width, height))
    fg_writer = cv2.VideoWriter(fg_video_path, fourcc, fps, (width, height))
    
    # First pass: learn the background
    frames_for_learning = min(sample_frames, frame_count)
    
    for i in range(frames_for_learning):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Apply background subtraction to update model
        fg_mask = bg_subtractor.apply(frame)
    
    # Reset video to beginning
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # Second pass: extract background and foreground
    frames_interval = int(fps * interval) if save_frames else 0
    bg_frames = []
    fg_frames = []
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Apply background subtraction
        fg_mask = bg_subtractor.apply(frame, learningRate=0)
        
        # Get the background image
        bg_image = bg_subtractor.getBackgroundImage()
        
        # Create foreground by applying the mask
        fg_image = cv2.bitwise_and(frame, frame, mask=(fg_mask > 0).astype(np.uint8))
        
        # Write frames to output videos
        bg_writer.write(bg_image)
        fg_writer.write(fg_image)
        
        # Save sample frames if requested
        if save_frames and frames_interval > 0 and frame_idx % frames_interval == 0:
            # Save background frame
            bg_frame_path = os.path.join(output_dir, f"bg_frame_{frame_idx:04d}.jpg")
            cv2.imwrite(bg_frame_path, bg_image)
            bg_frames.append({
                "frame_index": frame_idx,
                "path": bg_frame_path
            })
            
            # Save foreground frame
            fg_frame_path = os.path.join(output_dir, f"fg_frame_{frame_idx:04d}.jpg")
            cv2.imwrite(fg_frame_path, fg_image)
            fg_frames.append({
                "frame_index": frame_idx,
                "path": fg_frame_path
            })
        
        frame_idx += 1
    
    # Release resources
    cap.release()
    bg_writer.release()
    fg_writer.release()
    
    return {
        "video_path": video_path,
        "background_video": bg_video_path,
        "foreground_video": fg_video_path,
        "background_frames": bg_frames,
        "foreground_frames": fg_frames,
        "output_directory": output_dir
    } 