from app.common.resource_manager.resource_manager import ResourceManager
from typing import List, Dict, Any, Tuple
import cv2
import numpy as np
import os



def detect_scenes(resource_manager: ResourceManager, threshold: float = 30.0, min_scene_length: int = 15) -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
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
    
    # add last frame as a boundary
    scene_boundaries.append(frame_count)

    # Calculate scene information
    scene_images = []
    scene_info = []
    
    # Reset video position    
    for i, boundary in enumerate(scene_boundaries):
        start_frame_index = 0 if i == 0 else scene_boundaries[i-1]
        end_frame_index = boundary - 1
        mid_frame_index = (start_frame_index + end_frame_index) // 2

        indexes = [start_frame_index, mid_frame_index, end_frame_index]
        for index in indexes:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            _, frame = cap.read()
            scene_images.append(frame)        
        
        # Add scene details with simplified output
        scene_info.append({
            "start_time": start_frame_index / fps if fps > 0 else 0,
            "end_time": end_frame_index / fps if fps > 0 else 0,
            "duration": (end_frame_index - start_frame_index) / fps if fps > 0 else 0
        })
    
    return scene_images, scene_info