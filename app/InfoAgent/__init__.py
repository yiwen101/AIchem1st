"""
InfoAgent module implementing the IVideoAgent interface.

This module provides an InfoAgent class that focuses on extracting and utilizing
information from various sources like YouTube metadata, video transcripts, and
analyzing key frames to provide comprehensive answers.
"""

import os
import atexit
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

from app.common.temporal_query.temporal_query import (
    caption_based_analyze_temporal_query,
    most_important_image_based_temporal_query
)
from app.common.utils.youtube import get_youtube_video_info
from app.hypothesis_based_agent import QueryVisionLLMResponseHint
from app.model.interface import IVideoAgent
from app.model.structs import (
    ParquetFileRow, QueryVisionLLMResponseWithExplanation, 
    VisionModelRequest, YoutubeVideoInfo
)
from app.common.resource_manager.resource_manager import ResourceManager
from app.common.monitor import logger
from app.common.llm.openai import DEFAULT_SYSTEM_PROMPT, query_vision_llm
from pydantic import BaseModel

class InfoAgentQueryVisionLLMNoteResponse(BaseModel):
    description: str
    information_for_answer: str
    index_of_most_important_image: int

class InfoAgentQueryVisionLLMLastNoteResponse(BaseModel):
    description: str
    information_for_answer: str
    

class InfoAgent(IVideoAgent):
    """
    InfoAgent for comprehensive video analysis using information from multiple sources.
    
    This agent leverages information from YouTube metadata, transcripts, and 
    frame analysis to provide rich, context-aware answers to questions about videos.
    """
    
    def __init__(self, model: str = "gpt-4o-mini", display: bool = False):
        """
        Initialize the InfoAgent.
        
        Args:
            model_type: The LLM model to use for vision queries
            display: Whether to display images during analysis
        """
        self.model = model
        self.display = display
        self.resource_manager = ResourceManager()
        self.youtube_video_persistence_path = "persistence/youtube_video_info"
        if not os.path.exists(self.youtube_video_persistence_path):
            os.makedirs(self.youtube_video_persistence_path)        
        self.info_agent_persistence_path = "persistence/InfoAgent"
        if not os.path.exists(self.info_agent_persistence_path):
            os.makedirs(self.info_agent_persistence_path)
        # Register cleanup handler
        atexit.register(self._cleanup_resources)
    
    def _cleanup_resources(self):
        """Clean up video resources upon exit."""
        self.resource_manager.cleanup()
    
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
            metadata = self.resource_manager.load_video(video_path)
            logger.log_info(f"Loaded video {video_id} - Duration: {metadata['duration']:.2f}s, Resolution: {metadata['width']}x{metadata['height']}")
            return True
        except Exception as e:
            logger.log_error(f"Error loading video {video_id}: {str(e)}")
            return False
    
    def generate_hint_prompt(self, query: str, video_info: YoutubeVideoInfo) -> str:
        if not video_info.is_valid():
            return ""
        file_path = f"{self.youtube_video_persistence_path}/{self.row.video_id}.json"
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                queryVideoLLMResponse = QueryVisionLLMResponseHint.model_validate_json(f.read())
                return queryVideoLLMResponse.to_prompt()
        
        prompt = f"Please proces the following information about the youtube video. First, organise the information and try describe what the video is expected to be about. Next, Generate a hint prompt to help the LLM to answer the question: {query}"
        prompt += f"\nVideo information: {video_info.to_prompt()}"
        request = VisionModelRequest(prompt, [], response_class=QueryVisionLLMResponseHint)
        response = query_vision_llm(request, model=self.model, display=self.display, system_prompt=DEFAULT_SYSTEM_PROMPT)
        with open(file_path, "w") as f:
            f.write(response.model_dump_json())
        return response.to_prompt()
    
    def get_new_start_and_end_time(self, start: float, end: float, center_time: float) -> Tuple[float, float]:
        one_eighth_duration = (end - start) / 8
        new_start = max(0, center_time - one_eighth_duration)
        new_end = new_start + one_eighth_duration * 2
        return new_start, new_end
    
    def load_or_create_note(self, video_id: str) -> str:
        file_path = f"{self.info_agent_persistence_path}/{video_id}.txt"
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                return f.read()
        start = 0
        end = self.current_video_duration
        max_frames = [16,8,4]
        min_frame_intervals = [1,0.5,0.5]
        resp_prompts = []
        for max_frame, min_frame_interval in zip(max_frames, min_frame_intervals):
            is_last_iteration = (max_frame == max_frames[-1] and min_frame_interval == min_frame_intervals[-1])
            remaining_duration = end - start
            frame_number = min(max_frame, int(remaining_duration / min_frame_interval))
            frames, times = self.resource_manager.extract_frames_between(frame_number, start, end)
            if is_last_iteration:
                prompt = f"""
These {frame_number} images are extracted from {start} to {end} seconds of the video of duration {self.current_video_duration} seconds. 

Based on all these images, please extract relevant information for answering the following question in consistent English sentences: {self.query}"
First descript what you see in the video. Then note down relevant information for answering the question.
"""
                request = VisionModelRequest(prompt, frames, response_class=InfoAgentQueryVisionLLMLastNoteResponse)
                response = query_vision_llm(request, model=self.model, display=self.display, system_prompt=DEFAULT_SYSTEM_PROMPT)
                resp_prompt = f"""
Response from querying llm with {frame_number} images between {start} and {end} seconds of the video of duration {self.current_video_duration} seconds.
Please note that the image detail level is HIGH for this query, so the description of details is more accurate and convincing.

Description: {response.description}
Information for answer: {response.information_for_answer}
------------------------------------------------------------------------
"""
            else:
                prompt = f"""
These {frame_number} images are extracted from {start} to {end} seconds of the video of duration {self.current_video_duration} seconds. 

Based on all these images, please extract relevant information for answering the following question in consistent English sentences: {self.query}"
First descript what you see in the video. Then note down relevant information for answering the question. Lastly, pick the index of the image (1 to {frame_number}) that is most important for answering the question.
"""
            request = VisionModelRequest(prompt, frames, response_class=InfoAgentQueryVisionLLMNoteResponse)
            response = query_vision_llm(request, model=self.model, display=self.display, system_prompt=DEFAULT_SYSTEM_PROMPT)
            time_of_most_important_image = times[response.index_of_most_important_image - 1]

            resp_prompt = f"""
Response from querying llm with {frame_number} images between {start} and {end} seconds of the video of duration {self.current_video_duration} seconds.

Description: {response.description}
Information for answer: {response.information_for_answer}
time of most important image: {time_of_most_important_image}
------------------------------------------------------------------------
"""   
            resp_prompts.append(resp_prompt)
            start, end = self.get_new_start_and_end_time(start, end, time_of_most_important_image)
        
        file_content = "\n".join(resp_prompts)
        with open(file_path, "w") as f:
            f.write(file_content)
        return file_content
    
    def get_answer(self, row: ParquetFileRow) -> str:
        """
        Get an answer to a question about a video.
        
        Args:
            row: ParquetFileRow containing video information and query
            
        Returns:
            Answer to the query
        """
        # Set current query in the resource manager
        self.resource_manager.load_video_from_query(row)
        self.row = row
        self.query = row.question
        self.current_video_duration = float(row.duration)
        self.youtube_info_prompt = self.generate_hint_prompt(row.question, get_youtube_video_info(row))

        note = self.load_or_create_note(row.video_id)

        prompt = f"""
With reference to the following information that are extracted by making queries in decreasing order of scope, but increasing order of detail, try answer the following question: {self.query}

Information:
{note}
"""
        request = VisionModelRequest(prompt, [])
        response = query_vision_llm(request, model=self.model, display=self.display, system_prompt=DEFAULT_SYSTEM_PROMPT)
        return response.answer

    
    def get_agent_name(self) -> str:
        """
        Get the name of the agent.
        
        Returns:
            The agent name
        """
        return "InfoAgent"
    
    def get_system_prompt(self) -> str:
       return DEFAULT_SYSTEM_PROMPT + f"{self.youtube_info_prompt}"