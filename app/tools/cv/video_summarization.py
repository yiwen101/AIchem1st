"""
Video summarization tool.

This module provides functionality to summarize videos by extracting key frames
and generating captions for them.
"""

import os
from typing import Dict, Any, List, Optional

from app.tools.base_tool import BaseTool, ToolParameter, ToolParameterType
from app.tools.tool_manager import register_cv_tool
from app.tools.cv.video_frames import ExtractFramesTool
from app.tools.cv.image_captioning import CaptionImageTool


@register_cv_tool
class SummarizeVideoTool(BaseTool):
    """Tool for summarizing video content through frame extraction and captioning."""
    
    name = "summarize_video"
    description = "Extract key frames from a video and generate captions to summarize its content."
    parameters = [
        ToolParameter(
            name="video_path",
            type=ToolParameterType.STRING,
            description="Path to the video file",
            required=True
        ),
        ToolParameter(
            name="output_dir",
            type=ToolParameterType.STRING,
            description="Directory to save output (if not provided, a directory based on the video name will be created)",
            required=False,
            default=None
        ),
        ToolParameter(
            name="frame_interval",
            type=ToolParameterType.FLOAT,
            description="Interval in seconds between extracted frames",
            required=False,
            default=5.0
        ),
        ToolParameter(
            name="max_frames",
            type=ToolParameterType.INTEGER,
            description="Maximum number of frames to extract and caption",
            required=False,
            default=20
        ),
        ToolParameter(
            name="caption_prompt",
            type=ToolParameterType.STRING,
            description="Prompt for the captioning model",
            required=False,
            default="Describe what is happening in this frame from the video. Focus on actions, objects, and scene context."
        )
    ]
    
    @classmethod
    def execute(cls,
                video_path: str,
                output_dir: Optional[str] = None,
                frame_interval: float = 5.0,
                max_frames: int = 20,
                caption_prompt: str = "Describe what is happening in this frame from the video. Focus on actions, objects, and scene context."
                ) -> Dict[str, Any]:
        """
        Summarize a video by extracting key frames and generating captions.
        
        Args:
            video_path: Path to the video file
            output_dir: Directory to save output (if None, generated based on video name)
            frame_interval: Interval in seconds between extracted frames
            max_frames: Maximum number of frames to extract and caption
            caption_prompt: Prompt for the captioning model
            
        Returns:
            Dictionary with video information, extracted frames, and captions
            
        Raises:
            ValueError: If the video cannot be opened or processed
        """
        # Create output directory if it doesn't exist
        if not output_dir:
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            output_dir = f"{video_name}_summary"
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Step 1: Extract frames from the video
        frame_result = ExtractFramesTool.execute(
            video_path=video_path,
            output_dir=output_dir,
            interval=frame_interval,
            max_frames=max_frames
        )
        
        # Step 2: Generate captions for each frame
        captioned_frames = []
        
        for idx, frame_info in enumerate(frame_result["frames"]):
            frame_path = frame_info["path"]
            
            try:
                # Generate caption for the frame
                caption_result = CaptionImageTool.execute(
                    image_path=frame_path,
                    prompt=caption_prompt
                )
                
                # Add caption information to the frame data
                captioned_frame = {
                    **frame_info,
                    "caption": caption_result["caption"]
                }
                
                captioned_frames.append(captioned_frame)
                
            except Exception as e:
                captioned_frames.append({
                    **frame_info,
                    "caption": f"Error generating caption: {str(e)}",
                    "error": str(e)
                })
        
        # Step 3: Create a summary of the video
        # Generate timestamps as time ranges (start-end for each frame)
        timestamps = []
        for i, frame in enumerate(captioned_frames):
            time_pos = frame["time_position"]
            
            # For the last frame, estimate the end time
            if i == len(captioned_frames) - 1:
                end_time = min(time_pos + frame_interval, frame_result["duration"])
            else:
                end_time = captioned_frames[i+1]["time_position"]
            
            timestamps.append({
                "start": time_pos,
                "end": end_time,
                "caption": frame["caption"]
            })
        
        # Create a high-level summary using the captions
        summary = "Video Summary:\n\n"
        for ts in timestamps:
            start_formatted = format_timestamp(ts["start"])
            end_formatted = format_timestamp(ts["end"])
            summary += f"[{start_formatted} - {end_formatted}]: {ts['caption']}\n\n"
        
        # Save summary to a text file
        summary_path = os.path.join(output_dir, "video_summary.txt")
        with open(summary_path, "w") as f:
            f.write(summary)
        
        return {
            "video_path": video_path,
            "video_info": {
                "duration": frame_result["duration"],
                "fps": frame_result["fps"],
                "width": frame_result["width"],
                "height": frame_result["height"]
            },
            "frames": captioned_frames,
            "timestamps": timestamps,
            "summary_text": summary,
            "summary_file": summary_path,
            "output_directory": output_dir
        }


def format_timestamp(seconds: float) -> str:
    """
    Format seconds into HH:MM:SS format.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted timestamp
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    else:
        return f"{minutes:02d}:{seconds:02d}" 