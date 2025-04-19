"""
NaiveAgent module implementing the IVideoAgent interface.

This module provides a NaiveAgent class that takes a simple approach by
extracting 10 frames from the video and asking GPT-4o-mini directly.
"""

import os
import atexit
import numpy as np
from typing import List, Dict, Any

from app.model.interface import IVideoAgent
from app.model.structs import ParquetFileRow, VisionModelRequest
from app.common.resource_manager import resource_manager
from app.common.monitor import logger
from app.common.llm.openai import query_vision_llm

class NaiveAgent(IVideoAgent):
    """
    NaiveAgent implements the IVideoAgent interface using a simple approach.
    
    This agent extracts 10 frames evenly distributed across the video and
    sends them to GPT-4o-mini to answer the question directly.
    """
    
    def __init__(self, num_frames: int = 10, model: str = "gpt-4o-mini"):
        """
        Initialize the NaiveAgent.
        
        Args:
            num_frames: Number of frames to extract from the video
            model: The OpenAI model to use for answering questions
        """
        self.num_frames = num_frames
        self.model = model
        
        # Ensure output directories exist
        os.makedirs("videos", exist_ok=True)
        os.makedirs("app/tools/output/naive_agent", exist_ok=True)
        
        # Register cleanup function
        atexit.register(self._cleanup_resources)
    
    def _cleanup_resources(self):
        """Clean up video resources on exit."""
        logger.log_info("Cleaning up resources...")
        resource_manager.cleanup()
    
    def _preload_video(self, video_id: str) -> bool:
        """
        Preload a video into the resource manager.
        
        Args:
            video_id: ID of the video to load
            
        Returns:
            True if the video was loaded successfully, False otherwise
        """
        video_path = f"videos/{video_id}.mp4"
        
        # Check if video exists
        if not os.path.exists(video_path):
            logger.log_warning(f"Video file not found: {video_path}")
            return False
        
        try:
            # Load video into resource manager
            metadata = resource_manager.load_video(video_path)
            logger.log_info(f"Loaded video {video_id} - Duration: {metadata['duration']:.2f}s, Resolution: {metadata['width']}x{metadata['height']}")
            return True
        except Exception as e:
            logger.log_error(f"Error loading video {video_id}: {str(e)}")
            return False
    
    def get_answer(self, row: ParquetFileRow) -> str:
        """
        Get an answer to a question about a video.
        
        Args:
            row: ParquetFileRow containing the question and video information
            
        Returns:
            The answer to the question
        """
        # Load the video
        logger.log_info(f"Processing query {row.qid} for video {row.video_id}")
        if not self._preload_video(row.video_id):
            logger.log_error(f"Could not load video for {row.qid}")
            return "I couldn't load the video to answer this question."
        
        try:
            # Extract frames from the video using the resource manager
            duration = float(row.duration)
            frames = resource_manager.extract_frames_between(
                num_frames=self.num_frames,
                end_time=duration,
                save_frames=True,
                tool_name="naive_agent"
            )
            
            if not frames:
                logger.log_error(f"No frames extracted for video {row.video_id}")
                return "I couldn't extract frames from the video to answer this question."
            
            # Construct prompt with the question
            prompt = f"""I'm going to show you {len(frames)} frames from a video. Please answer the following question based on these frames:
            
Question: {row.question}

Provide a concise answer. If you can't determine the answer from these frames, explain why.
"""
            
            # Create a vision request with all frames
            request = VisionModelRequest(prompt, frames, high_detail=False)
            
            # Query GPT-4o-mini
            logger.log_info(f"Querying {self.model} with {len(frames)} frames")
            response = query_vision_llm(request, model=self.model)
            
            logger.log_info(f"Answer: {response}")
            return response.answer
            
        except Exception as e:
            logger.log_error(f"Error processing query {row.qid}: {str(e)}")
            return f"Error processing the query: {str(e)}"
    
    def get_agent_name(self) -> str:
        """
        Get the name of this agent.
        
        Returns:
            The name of the agent
        """
        return "NaiveAgent"

# Export the NaiveAgent class
__all__ = ["NaiveAgent"] 