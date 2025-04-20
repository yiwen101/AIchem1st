"""
LLM-based motion detection using GPT-4o's vision capabilities.

This module provides a tool that samples frames from a video at regular intervals,
then uses GPT-4o to analyze and describe the motion between consecutive frames.
"""

import os
import cv2
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import time
import json
import matplotlib.pyplot as plt
from PIL import Image

from app.tools.toolImpl.base_tool import BaseTool, ToolParameter, ToolParameterType
from app.tools.tool_manager import register_tool
from app.common.resource_manager.resource_manager import resource_manager
from app.common.monitor import logger
from app.common.llm.openai import query_vision_llm_single_image


@register_tool
class LLMBasedMotionDetectionTool(BaseTool):
    """Tool for detecting motion between video frames using GPT-4o's vision capabilities."""
    
    name = "llm_motion_detection"
    description = "Detect and describe motion between video frames using GPT-4o's vision model."
    parameters = [
        ToolParameter(
            name="start_time",
            type=ToolParameterType.FLOAT,
            description="Start time in seconds for motion analysis",
            required=True
        ),
        ToolParameter(
            name="end_time",
            type=ToolParameterType.FLOAT,
            description="End time in seconds for motion analysis",
            required=True
        ),
        ToolParameter(
            name="frame_interval",
            type=ToolParameterType.FLOAT,
            description="Interval between sampled frames in seconds",
            required=False,
            default=1.0
        ),
        ToolParameter(
            name="detailed",
            type=ToolParameterType.BOOLEAN,
            description="Whether to provide detailed motion descriptions",
            required=False,
            default=True
        )
    ]
    
    @classmethod
    def execute(cls, start_time: float, end_time: float, 
               frame_interval: float = 1.0,
               detailed: bool = True) -> Dict[str, Any]:
        """
        Detect and describe motion between video frames using GPT-4o.
        
        Args:
            start_time: Start time in seconds
            end_time: End time in seconds
            frame_interval: Interval between sampled frames in seconds
            detailed: Whether to provide detailed descriptions
            
        Returns:
            Dictionary with motion analysis results
        """
        # Store original requested time range
        requested_start_time = start_time
        requested_end_time = end_time
        
        # Get active video
        cap, metadata = resource_manager.get_active_video()
        
        # Get video properties
        fps = metadata["fps"]
        frame_count = metadata["frame_count"]
        width = metadata["width"]
        height = metadata["height"]
        video_name = metadata["video_name"]
        
        # Calculate frames
        start_frame = int(start_time * fps)
        end_frame = min(int(end_time * fps), frame_count - 1)
        
        if start_frame >= end_frame:
            logger.log_error(f"Invalid time range: start_time={start_time}, end_time={end_time}")
            return {
                "error": f"Invalid time range: start_time={start_time}, end_time={end_time}",
                "time_range": f"{requested_start_time:.1f}s to {requested_end_time:.1f}s"
            }
        
        # Calculate frame sampling based on interval
        frame_step = int(frame_interval * fps)
        frame_step = max(1, frame_step)  # Ensure at least 1 frame step
        
        logger.log_info(f"Analyzing motion using GPT-4o from {start_time:.2f}s to {end_time:.2f}s "
                      f"with {frame_interval}s intervals ({frame_step} frames)")
        
        # Sample frames
        frame_times = []
        frames = []
        
        # Set output directory for frames
        output_dir = f"app/tools/output/llm_motion/{video_name}/{int(start_time)}_{int(end_time)}"
        os.makedirs(output_dir, exist_ok=True)
        
        for frame_idx in range(start_frame, end_frame + 1, frame_step):
            # Get frame at position
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                break
                
            # Calculate time for this frame
            frame_time = frame_idx / fps
            
            # Save frame information
            frame_times.append(frame_time)
            frames.append(frame)
            
            # Save the frame as an image
            frame_path = os.path.join(output_dir, f"frame_{frame_time:.2f}.jpg")
            cv2.imwrite(frame_path, frame)
            
            logger.log_info(f"Sampled frame at time {frame_time:.2f}s")
        
        if len(frames) < 2:
            logger.log_error(f"Not enough frames sampled for motion analysis (only {len(frames)} frames)")
            return {
                "error": f"Not enough frames sampled for motion analysis. Need at least 2 frames.",
                "time_range": f"{requested_start_time:.1f}s to {requested_end_time:.1f}s"
            }
        
        # Analyze motion between consecutive frames
        motion_descriptions = []
        frame_pairs = []
        
        logger.log_info(f"Analyzing motion between {len(frames)} frames using GPT-4o")
        
        for i in range(len(frames) - 1):
            frame1 = frames[i]
            frame2 = frames[i + 1]
            time1 = frame_times[i]
            time2 = frame_times[i + 1]
            
            # Create a side-by-side comparison image
            combined_img = np.hstack((frame1, frame2))
            comparison_path = os.path.join(output_dir, f"comparison_{time1:.2f}_to_{time2:.2f}.jpg")
            cv2.imwrite(comparison_path, combined_img)
            
            # Create a prompt for GPT-4o
            if detailed:
                prompt = (f"Analyze the motion between these two consecutive video frames taken {frame_interval:.1f} seconds apart. "
                         f"The first frame is from {time1:.2f}s and the second frame is from {time2:.2f}s into the video. "
                         f"Describe in detail:\n"
                         f"1. What objects have moved between the two timestamps\n"
                         f"2. The direction and approximate distance of movement\n"
                         f"3. Any changes in object appearance or position\n"
                         f"4. Any new objects that appear or disappear\n"
                         f"Focus only on significant motion and ignore minor changes that might be due to lighting or camera artifacts.")
            else:
                prompt = (f"Describe the motion and changes between these two consecutive video frames taken {frame_interval:.1f} seconds apart. "
                         f"The first frame is from {time1:.2f}s and the second frame is from {time2:.2f}s into the video. "
                         f"Keep your answer brief and focus only on significant movement.")
            
            # Query GPT-4o for motion description
            try:
                description = query_vision_llm_single_image(combined_img, prompt)
                
                # Save the result
                motion_descriptions.append({
                    "start_time": time1,
                    "end_time": time2,
                    "description": description,
                    "comparison_image": comparison_path
                })
                
                frame_pairs.append((time1, time2))
                
                logger.log_info(f"Analyzed motion from {time1:.2f}s to {time2:.2f}s")
            except Exception as e:
                logger.log_error(f"Error analyzing motion from {time1:.2f}s to {time2:.2f}s: {str(e)}")
                motion_descriptions.append({
                    "start_time": time1,
                    "end_time": time2,
                    "error": str(e),
                    "comparison_image": comparison_path
                })
        
        # Create a visualization showing all frame pairs with their descriptions
        try:
            visualization_path = os.path.join(output_dir, "motion_summary.jpg")
            
            # Create a figure with subplots for each frame pair
            fig, axes = plt.subplots(len(frame_pairs), 1, figsize=(12, 5 * len(frame_pairs)))
            
            # Handle the case with only one pair
            if len(frame_pairs) == 1:
                axes = [axes]
            
            for i, ((time1, time2), motion_desc) in enumerate(zip(frame_pairs, motion_descriptions)):
                # Load the comparison image
                img = cv2.imread(motion_desc["comparison_image"])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Display the image
                axes[i].imshow(img)
                
                # Add title with timestamp range
                axes[i].set_title(f"Motion from {time1:.2f}s to {time2:.2f}s")
                
                # Add description as text below the image
                desc_text = motion_desc.get("description", "No description available")
                
                # Truncate long descriptions
                if len(desc_text) > 200 and not detailed:
                    desc_text = desc_text[:197] + "..."
                
                axes[i].set_xlabel(desc_text, wrap=True)
                axes[i].xaxis.label.set_fontsize(10)
                axes[i].xaxis.label.set_linespacing(0.8)
                
                # Remove ticks
                axes[i].set_xticks([])
                axes[i].set_yticks([])
            
            # Add a title to the figure
            fig.suptitle(f"Motion Analysis: {requested_start_time:.1f}s to {requested_end_time:.1f}s", 
                       fontsize=16)
            
            # Adjust layout
            plt.tight_layout(rect=[0, 0, 1, 0.97])
            
            # Save the figure
            plt.savefig(visualization_path)
            plt.close(fig)
            
            logger.log_info(f"Created motion summary visualization at {visualization_path}")
        except Exception as e:
            logger.log_error(f"Error creating visualization: {str(e)}")
            visualization_path = None
        
        # Create a summary of the motion across the entire clip
        try:
            # Combine all descriptions
            all_descriptions = [m.get("description", "") for m in motion_descriptions if "description" in m]
            
            # Create a prompt for GPT-4o to synthesize the observations
            if all_descriptions:
                summary_prompt = (
                    f"I have analyzed motion between consecutive frames in a video from {requested_start_time:.1f}s "
                    f"to {requested_end_time:.1f}s. Here are the individual observations:\n\n" + 
                    "\n\n".join([f"From {motion_descriptions[i]['start_time']:.2f}s to {motion_descriptions[i]['end_time']:.2f}s: {desc}" 
                               for i, desc in enumerate(all_descriptions)]) +
                    "\n\nBased on these observations, provide a concise summary of the overall motion and activity in this video clip."
                )
                
                overall_summary = query_vision_llm(None, summary_prompt)
            else:
                overall_summary = "Could not generate a summary due to missing motion descriptions."
        except Exception as e:
            logger.log_error(f"Error generating overall summary: {str(e)}")
            overall_summary = f"Error generating summary: {str(e)}"
        
        # Create the result
        result = {
            "time_range": f"{requested_start_time:.1f}s to {requested_end_time:.1f}s",
            "frame_interval": frame_interval,
            "frames_analyzed": len(frames),
            "motion_segments": len(motion_descriptions),
            "overall_summary": overall_summary,
            "motion_descriptions": motion_descriptions
        }
        
        # Add visualization path if available
        if visualization_path:
            result["visualization"] = visualization_path
        
        return result 