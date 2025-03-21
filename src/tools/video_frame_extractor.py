import cv2
import os
import base64
from io import BytesIO
from PIL import Image
from typing import Dict, Any, List, Optional
from tools.base_tool import BaseTool

class VideoFrameExtractor(BaseTool):
    """Tool to extract frames from a video at specific timestamps."""
    
    @property
    def description(self) -> str:
        return "Extracts frames from a video file at specific timestamps or intervals."
    
    @property
    def input_schema(self) -> Dict[str, Dict[str, Any]]:
        return {
            "video_path": {
                "type": "string",
                "description": "Path to the video file",
                "required": True
            },
            "timestamps": {
                "type": "array",
                "description": "List of timestamps (in seconds) to extract frames from",
                "required": False
            },
            "frame_count": {
                "type": "integer",
                "description": "Number of frames to extract if timestamps not provided",
                "required": False,
                "default": 1
            },
            "interval": {
                "type": "number",
                "description": "Interval between frames if frame_count > 1",
                "required": False
            }
        }
    
    @property
    def output_schema(self) -> Dict[str, Any]:
        return {
            "frames": {
                "type": "object",
                "description": "Extracted frames as base64 encoded strings with timestamps as keys"
            },
            "video_info": {
                "type": "object",
                "description": "Information about the video including fps, duration, and total frames"
            }
        }
    
    def execute(self, video_path: str, timestamps: Optional[List[float]] = None, 
                frame_count: int = 1, interval: Optional[float] = None) -> Dict[str, Any]:
        """
        Extract frames from a video.
        
        Args:
            video_path (str): Path to the video file
            timestamps (list, optional): List of timestamps (in seconds) to extract
            frame_count (int, optional): Number of frames to extract if timestamps not provided
            interval (float, optional): Interval between frames if frame_count > 1
            
        Returns:
            dict: Extracted frames as base64 encoded strings with timestamps
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise Exception(f"Could not open video file: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration = total_frames / fps if fps > 0 else 0
        
        frames = {}
        
        if timestamps:
            # Extract frames at specific timestamps
            for ts in timestamps:
                if 0 <= ts <= video_duration:
                    frame_number = int(ts * fps)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                    ret, frame = cap.read()
                    
                    if ret:
                        frames[str(ts)] = self._encode_frame(frame)
        else:
            # Extract evenly spaced frames
            if interval is None and frame_count > 1:
                interval = video_duration / (frame_count)
            
            if frame_count == 1:
                # Just get the middle frame
                middle_frame = total_frames // 2
                cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
                ret, frame = cap.read()
                
                if ret:
                    frames["middle"] = self._encode_frame(frame)
            else:
                # Get frames at regular intervals
                for i in range(frame_count):
                    timestamp = i * interval
                    frame_number = int(timestamp * fps)
                    
                    if frame_number < total_frames:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                        ret, frame = cap.read()
                        
                        if ret:
                            frames[str(timestamp)] = self._encode_frame(frame)
        
        cap.release()
        
        return {
            "frames": frames,
            "video_info": {
                "fps": fps,
                "duration": video_duration,
                "total_frames": total_frames
            }
        }
    
    def _encode_frame(self, frame):
        """Convert a frame to base64 encoded string."""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(rgb_frame)
        
        # Save to BytesIO object
        buffered = BytesIO()
        pil_image.save(buffered, format="JPEG")
        
        # Encode to base64
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return img_str 