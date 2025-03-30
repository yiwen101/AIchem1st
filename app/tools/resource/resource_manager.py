"""
Resource manager for video processing.

This module provides a resource manager that loads and caches video frames,
and provides methods to extract frames from videos.
"""

import os
import cv2
import numpy as np
from typing import Dict, Any, Tuple, Optional
import time

class ResourceManager:
    """
    Resource manager for handling video resources.
    
    This class provides methods for loading videos, extracting frames,
    and saving images with annotations. It maintains only one active video
    at a time to conserve memory.
    """
    
    def __init__(self):
        """Initialize the resource manager."""
        self.active_video = None
        self.active_video_path = None
        self.active_video_metadata = None
        self.frame_cache = {}
        self.output_root = "app/tools/output"
        
        # Create output directories
        os.makedirs(self.output_root, exist_ok=True)
        os.makedirs(f"{self.output_root}/image_captioning", exist_ok=True)
        os.makedirs(f"{self.output_root}/object_detection", exist_ok=True)
        os.makedirs(f"{self.output_root}/scene_detection", exist_ok=True)
    
    def load_video(self, video_path: str) -> Dict[str, Any]:
        """
        Load a video and make it the active video.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dictionary with video metadata
            
        Raises:
            ValueError: If the video cannot be opened
        """
        # Close previous video if any
        if self.active_video is not None and self.active_video.isOpened():
            self.active_video.release()
            self.frame_cache.clear()
        
        # Open new video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Extract metadata
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0
        
        metadata = {
            "fps": fps,
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "duration": duration,
            "video_name": os.path.splitext(os.path.basename(video_path))[0]
        }
        
        # Set as active video
        self.active_video = cap
        self.active_video_path = video_path
        self.active_video_metadata = metadata
        
        return metadata
    
    def get_active_video(self) -> Tuple[cv2.VideoCapture, Dict[str, Any]]:
        """
        Get the currently active video.
        
        Returns:
            Tuple of (VideoCapture, metadata_dict)
            
        Raises:
            ValueError: If no video is active
        """
        if self.active_video is None or not self.active_video.isOpened():
            raise ValueError("No active video. Call load_video first.")
        
        return self.active_video, self.active_video_metadata
    
    def get_frame_at_time(self, time_seconds: float) -> Tuple[np.ndarray, int]:
        """
        Extract a frame at specified time from the active video.
        
        Args:
            time_seconds: Time in seconds (e.g., 70.45 for 70 seconds and 45 milliseconds)
            
        Returns:
            Tuple of (frame as numpy array, frame index)
            
        Raises:
            ValueError: If the frame cannot be extracted or no video is active
        """
        # Get active video
        cap, metadata = self.get_active_video()
        
        # Calculate frame index from time
        frame_index = int(time_seconds * metadata["fps"])
        
        # Check if frame is in bounds
        if frame_index < 0 or frame_index >= metadata["frame_count"]:
            raise ValueError(f"Time {time_seconds}s is out of range for video with duration {metadata['duration']}s")
        
        # Check if frame is in cache
        cache_key = f"{frame_index}"
        if cache_key in self.frame_cache:
            return self.frame_cache[cache_key], frame_index
        
        # Set position to the specified frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        
        # Read the frame
        ret, frame = cap.read()
        if not ret:
            raise ValueError(f"Could not read frame at time {time_seconds}s (frame {frame_index})")
        
        # Cache the frame
        self.frame_cache[cache_key] = frame
        return frame, frame_index
    
    def save_image(self, 
                   image: np.ndarray, 
                   time_seconds: float, 
                   tool_name: str, 
                   suffix: str = "", 
                   subdirectory: str = "") -> str:
        """
        Save an image to the output directory.
        
        Args:
            image: Image as numpy array
            time_seconds: Time in seconds (e.g., 70.45 for 70 seconds and 45 milliseconds)
            tool_name: Name of the tool (image_captioning, object_detection, scene_detection)
            suffix: Optional suffix for the filename
            subdirectory: Optional subdirectory within the tool directory
            
        Returns:
            Path to the saved image
        """
        # Ensure we have an active video
        if self.active_video_metadata is None:
            raise ValueError("No active video. Call load_video first.")
            
        # Get video name from metadata
        video_name = self.active_video_metadata["video_name"]
        
        # Create directory for this video if it doesn't exist
        output_dir = f"{self.output_root}/{tool_name}/{video_name}"
        if subdirectory:
            output_dir = f"{output_dir}/{subdirectory}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Format time as SS_MS
        time_formatted = f"{int(time_seconds):02d}_{int((time_seconds % 1) * 1000):03d}"
        
        # Create filename
        filename = f"{time_formatted}{suffix}.jpg"
        output_path = os.path.join(output_dir, filename)
        
        # Save the image
        cv2.imwrite(output_path, image)
        return output_path
    
    def save_captioned_image(self, 
                             image: np.ndarray, 
                             time_seconds: float, 
                             caption: str, 
                             tool_name: str = "image_captioning",
                             subdirectory: str = "") -> str:
        """
        Save an image with caption overlaid.
        
        Args:
            image: Image as numpy array
            time_seconds: Time in seconds (e.g., 70.45 for 70 seconds and 45 milliseconds)
            caption: Caption text to overlay
            tool_name: Name of the tool directory
            subdirectory: Optional subdirectory within the tool directory
            
        Returns:
            Path to the saved image
        """
        # Create a copy of the image to avoid modifying the original
        img_with_caption = image.copy()
        
        # Add caption text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_color = (255, 255, 255)  # White
        font_thickness = 2
        
        # Wrap text to fit image width
        height, width = img_with_caption.shape[:2]
        max_width = width - 20  # Margin
        char_width = int(font_scale * 20)  # Approximate char width
        chars_per_line = max(1, max_width // char_width)
        
        # Split caption into lines
        words = caption.split()
        lines = []
        current_line = ""
        
        for word in words:
            if len(current_line + word) + 1 <= chars_per_line:
                current_line += " " + word if current_line else word
            else:
                lines.append(current_line)
                current_line = word
        
        if current_line:
            lines.append(current_line)
        
        # Draw semi-transparent background for text
        overlay = img_with_caption.copy()
        bg_height = len(lines) * 30 + 20
        cv2.rectangle(overlay, (0, 0), (width, bg_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, img_with_caption, 0.4, 0, img_with_caption)
        
        # Draw text
        y_position = 30
        for line in lines:
            cv2.putText(img_with_caption, line, (10, y_position), font, font_scale, font_color, font_thickness)
            y_position += 30
        
        # Save the captioned image
        return self.save_image(img_with_caption, time_seconds, tool_name, 
                              suffix="_captioned", subdirectory=subdirectory)
    
    def cleanup(self):
        """Release all video resources."""
        if self.active_video is not None and self.active_video.isOpened():
            self.active_video.release()
            self.active_video = None
            self.active_video_path = None
            self.active_video_metadata = None
        
        # Clear frame cache
        self.frame_cache.clear()
        
        print("ResourceManager: Released video resources")

# Create a singleton instance
resource_manager = ResourceManager() 