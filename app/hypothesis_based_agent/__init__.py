"""
HypothesisBasedAgent module implementing the IVideoAgent interface.

This agent analyzes videos by forming hypotheses first, then evaluating them against the video.
"""

import os
import atexit
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import json

from app.common.utils.youtube import get_youtube_video_info
from app.model.interface import IVideoAgent
from app.model.structs import ParquetFileRow, VisionModelRequest, YoutubeVideoInfo, QueryVisionLLMResponseWithExplanation
from app.common.resource_manager.resource_manager import ResourceManager
from app.common.monitor import logger
from app.common.llm.openai import DEFAULT_SYSTEM_PROMPT, query_vision_llm
from pydantic import BaseModel, Field

class YouTubeInfoResponse(BaseModel):
    youtube_video_description: str
    focus_prompt: str

class VideoDescriptionResponse(BaseModel):
    video_description: str
    significant_events: List[str]
    hint_prompt: str

    def to_prompt(self):
        description_text = self.video_description
        significant_events = ", ".join(self.significant_events)
        return f"Video description: {description_text}\nSignificant events: {significant_events}\nHint prompt: {self.hint_prompt}"
    
class PotentialAnswersResponse(BaseModel):
    potential_answers: List[str] = Field(..., description="List of potential answers to the query")
    reasoning: str

    def to_prompt(self):
        return f"Potential answers: {self.potential_answers}\nReasoning: {self.reasoning}"
    
class FinalAnswerResponse(BaseModel):
    answer: str
    explanation: str
    confidence: str  # high, medium, low

class QueryVisionLLMResponseHint(BaseModel):
    youtube_video_description: str
    hint_prompt: str

    def to_prompt(self):
        return f"\n{self.youtube_video_description}\nHint: {self.hint_prompt}"

class HypothesisBasedAgent(IVideoAgent):
    """
    HypothesisBasedAgent implements the IVideoAgent interface.
    
    This agent analyzes videos by:
    1. Getting a detailed video description (from YouTube info and/or video frames)
    2. Generating potential answers based on the description
    3. Evaluating these answers against the full video
    4. Providing a final answer with explanation
    """
    
    def __init__(self, model: str = "gpt-4o-mini", display: bool = False):
        """
        Initialize the HypothesisBasedAgent.
        
        Args:
            model: The OpenAI model to use for answering questions
            display: Whether to display images during analysis
        """
        self.model = model
        self.display = display
        self.resource_manager = ResourceManager()
        self.persistence_dir = "persistence/hypothesis_agent"
        self.current_video_dir = ""
        os.makedirs(self.persistence_dir, exist_ok=True)    
        
        # Ensure output directories exist
        os.makedirs("videos", exist_ok=True)
        
        # Register cleanup function
        atexit.register(self._cleanup_resources)
    
    def _cleanup_resources(self):
        """Clean up video resources on exit."""
        logger.log_info("Cleaning up resources...")
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
        self.current_video_dir = f"{self.persistence_dir}/{video_id}"
        os.makedirs(self.current_video_dir, exist_ok=True)
        
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
    
    def extract_full_video_frames(self) -> List[np.ndarray]:
        """
        Extract frames from the entire video.
        
        Returns:
            List of frames as numpy arrays
        """
        # Get video metadata
        _, metadata = self.resource_manager.get_active_video()
        video_duration = metadata['duration']
        
        # Extract frames evenly across the entire video - more frames for longer videos
        num_frames = min(20, int(video_duration / 1))
        frames, _ = self.resource_manager.extract_frames_between(
            num_frames=num_frames
        )
        
        return frames

    def generate_hint_prompt(self, query: str, video_info: YoutubeVideoInfo) -> str:
        if not video_info.is_valid():
            return ""
        prompt = f"Please proces the following information about the youtube video. First, organise the information and try describe what the video is expected to be about. Next, Generate a hint prompt to help the LLM to answer the question: {query}"
        prompt += f"\nVideo information: {video_info.to_prompt()}"
        request = VisionModelRequest(prompt, [], response_class=QueryVisionLLMResponseHint)
        response = query_vision_llm(request, model=self.model, display=self.display, system_prompt=self.get_system_prompt())
        return response
    
    def get_description(self, query: str, row: ParquetFileRow) -> Tuple[VideoDescriptionResponse, List[np.ndarray]]:
        """
        Get a detailed description of the video content, handling YouTube info processing.
        
        Args:
            query: The question to answer
            row: ParquetFileRow containing the video information
            
        Returns:
            Tuple of (video description, frames)
        """
        logger.log_info(f"Getting video description for query: {query}")
        
        # Step 1: Extract frames from the video
        frames = self.extract_full_video_frames()
        if not frames:
            logger.log_error("Failed to extract frames from video")
            raise ValueError("Failed to extract frames from video")

        # check whether the description file exists
        description_file = f"{self.current_video_dir}/description.json"
        if os.path.exists(description_file):
            with open(description_file, "r") as f:
                description_obj = json.load(f)
                description = VideoDescriptionResponse(**description_obj)
            return description, frames
        
        # Step 2: Get and organize YouTube information if available
        youtube_info = get_youtube_video_info(row)
        
        if youtube_info.is_valid():
            res = self.generate_hint_prompt(query, youtube_info)
            prompt = f"""
Analyze these frames extracted from a video and provide a detailed description for answering the following question: {query}

Following is some information about this video:
{res.to_prompt()}

Using both this information and what you can directly observe in these frames:
1. Improve the description of the video content
2. Improve the hint prompt to help the LLM to answer the question
3. Note significant events or actions that occur

Pay special attention to elements that might be relevant to answering this query: {query}
"""
        else:
            prompt = f"""
Analyze these frames extracted from a video and provide a detailed description for answering the following question: {query}

1. Provide a comprehensive description of the video content in chronological order
2. Note significant events or actions that occur
3. Generate a hint prompt to help the LLM to answer the question

Pay special attention to elements that might be relevant to answering this query: {query}
"""
        
        request = VisionModelRequest(
            query=prompt,
            images=frames,
            response_class=VideoDescriptionResponse
        )
        
        try:
            description = query_vision_llm(request, model=self.model, display=self.display)
            logger.log_info(f"Generated video description: {description.video_description[:100]}...")
            # save the description to the file
            with open(description_file, "w") as f:
                json.dump(description.model_dump(), f)
            return description, frames
        except Exception as e:
            logger.log_error(f"Error generating video description: {str(e)}")
            # Return a minimal description on error
            return VideoDescriptionResponse(
                video_description="Error generating description",
                significant_events=[]
            ), frames
    
    def get_hypotheses(self, query: str, description: VideoDescriptionResponse, frames: List[np.ndarray]) -> PotentialAnswersResponse:
        logger.log_info(f"Generating hypotheses for query: {query}")

        # check whether the hypotheses file exists
        hypotheses_file = f"{self.current_video_dir}/hypotheses.json"
        if os.path.exists(hypotheses_file):
            with open(hypotheses_file, "r") as f:
                hypotheses_obj = json.load(f)
                hypotheses = PotentialAnswersResponse(**hypotheses_obj)
            return hypotheses
        
        prompt = f"""
Based on these video frames and the following description, generate potential answers to this question: "{query}"
{description.to_prompt()}

Generate likely answer(s) to the question based on the video description and frames.
Make sure your answer are distinct from each other and based on different assumptions. It is ok to have only one suggested answer.
"""
        
        request = VisionModelRequest(
            query=prompt,
            images=frames,
            response_class=PotentialAnswersResponse
        )
        
        hypotheses = query_vision_llm(request, model=self.model, display=self.display)  
        # save the hypotheses to the file
        with open(hypotheses_file, "w") as f:
            json.dump(hypotheses.model_dump(), f)
        return hypotheses
       
    
    def get_final_answer(self, query: str, description: VideoDescriptionResponse, hypotheses: PotentialAnswersResponse, frames: List[np.ndarray]) -> FinalAnswerResponse:
        """
        Determine the final answer by evaluating hypotheses against the video.
        
        Args:
            query: The question to answer
            description: Video description
            hypotheses: Potential answers with reasoning
            frames: Frames from the video
            
        Returns:
            Final answer with explanation
        """
        logger.log_info(f"Determining final answer for query: {query}")
        
        # Format potential answers for the prompt        
        prompt = f"""
Please answer this question about a video: "{query}"

{description.to_prompt()}

Based on careful analysis of these video frames and the information provided, provide the best answer to the original question and a detailed explanation justifying your answer to the original question. Do not prelude the existence of likely answers.
"""
        
        request = VisionModelRequest(
            query=prompt,
            images=frames,
            response_class=FinalAnswerResponse
        )
        

        final_answer = query_vision_llm(request, model=self.model, display=self.display)
        logger.log_info(f"Final answer: {final_answer.answer}")
        logger.log_info(f"Confidence: {final_answer.confidence}")
        return final_answer
    
    def get_answer(self, row: ParquetFileRow) -> str:
        """
        Get an answer to a question about a video using a streamlined workflow.
        
        Args:
            row: ParquetFileRow containing the question and video information
            
        Returns:
            The answer to the question
        """
        # Load the video into resource manager
        logger.log_info(f"Processing query {row.qid} for video {row.video_id}")
        if not self._preload_video(row.video_id):
            logger.log_error(f"Could not load video for {row.qid}")
            return "I couldn't load the video to answer this question."
        
        try:
            query = row.question
            
            # Phase 1: Get video description (includes YouTube info processing)
            description, frames = self.get_description(query, row)
            
            '''
            # Phase 2: Generate hypotheses based on description
            hypotheses = self.get_hypotheses(query, description, frames)
            '''
            # Phase 3: Determine final answer based on description and hypotheses
            final_answer = self.get_final_answer(query, description, None, frames)
            
            # Return the final answer
            return f"{final_answer.answer} {final_answer.explanation}"
            
        except Exception as e:
            logger.log_error(f"Error in HypothesisBasedAgent.get_answer: {str(e)}")
            return f"Error analyzing the video: {str(e)}"
    
    def get_agent_name(self) -> str:
        """
        Get the name of this agent.
        
        Returns:
            The name of the agent
        """
        return f"HypothesisBasedAgent_{self.model}"
    
    def get_system_prompt(self) -> str:
        """
        Get the system prompt for the agent.
        
        Returns:
            The system prompt string
        """
        return DEFAULT_SYSTEM_PROMPT

# Export the HypothesisBasedAgent class
__all__ = ["HypothesisBasedAgent"] 