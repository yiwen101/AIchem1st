"""
L3Agent (Layered Learning Logic Agent) module implementing the IVideoAgent interface.

This module provides a L3Agent class that uses a multi-layered approach to video analysis,
progressively refining from whole video to important scenes to single images.
"""

import os
import atexit
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import json

from app.common.temporal_query.temporal_query import most_important_image_based_temporal_query
from app.common.utils.youtube import get_youtube_video_info
from app.model.interface import IVideoAgent
from app.model.structs import ParquetFileRow, QueryVisionLLMResponseWithExplanation, VisionModelRequest, YoutubeVideoInfo
from app.common.resource_manager.resource_manager import resource_manager
from app.common.monitor import logger
from app.common.llm.openai import DEFAULT_SYSTEM_PROMPT, query_vision_llm
from pydantic import BaseModel

class QueryVisionLLMResponseFullVideo(BaseModel):
    answer: str
    explanation: str
    description: str
    need_zoom_in_to_part_of_video: bool

class QueryVisionLLMResponseSegment(BaseModel):
    answer: str
    explanation: str
    description: str
    need_zoom_in_to_single_image_of_video: bool

class QueryVisionLLMResponseKeyImage(BaseModel):
    answer: str
    explanation: str
    description: str
    confidence: str

class QueryVisionLLMResponseHint(BaseModel):
    youtube_video_description: str
    hint_prompt: str

    def to_prompt(self):
        return f"\n{self.youtube_video_description}\nHint: {self.hint_prompt}"

class L3Agent(IVideoAgent):
    """
    L3Agent (Layered Learning Logic Agent) implements the IVideoAgent interface.
    
    This agent analyzes videos through progressive refinement:
    1. Analyze the entire video
    2. Identify most important image/segment
    3. Analyze the relevant segment
    4. Optionally analyze an extended segment
    5. Analyze the key image in high detail
    6. Synthesize all analyses into a final answer
    """
    
    def __init__(self, model: str = "gpt-4o-mini", display: bool = False):
        """
        Initialize the L3Agent.
        
        Args:
            model: The OpenAI model to use for answering questions
            display: Whether to display images during analysis
        """
        self.model = model
        self.display = display
        self.hint_prompt = ""
        
        # Ensure output directories exist
        os.makedirs("videos", exist_ok=True)
        os.makedirs("app/tools/output/l3_agent", exist_ok=True)
        
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
    
    def analyze_full_video(self, query: str) -> QueryVisionLLMResponseFullVideo:
        """
        Analyze the entire video to answer a query.
        
        Args:
            query: The question to answer
            
        Returns:
            Tuple of (answer, explanation)
        """
        logger.log_info(f"Analyzing full video for query: {query}")
        
        # Get video metadata
        _, metadata = resource_manager.get_active_video()
        video_duration = metadata['duration']
        
        # Extract frames evenly across the entire video
        num_frames = min(15, int(video_duration))
        frames, _ = resource_manager.extract_frames_between(
            num_frames=num_frames,
            save_frames=False
        )
        
        # Create the prompt
        prompt = f"These images are extracted to show the entire video from start to finish. Based on all these images, please answer the following question referring to the original video: {query}"
        prompt += "\nFirst descript what you see in the video. Then provide both a direct answer and a detailed explanation of your reasoning. If you think the answer is not clear, set need_zoom_in_to_part_of_video to True so that we can zoom in on the most important parts of the video and try again."
        #prompt += self.hint_prompt
        
        # Query the vision model
        request = VisionModelRequest(prompt, frames, response_class=QueryVisionLLMResponseFullVideo)
        try:
            response = query_vision_llm(request, model=self.model, display=self.display, system_prompt=self.get_system_prompt())
            logger.log_info(f"Full video analysis - Answer: {response.answer}")
            logger.log_info(f"Full video analysis - Explanation: {response.explanation[:100]}...")
            logger.log_info(f"Full video analysis - Need zoom in to part of video: {response.need_zoom_in_to_part_of_video}")
            return response
        except Exception as e:
            logger.log_error(f"Error in full video analysis: {str(e)}")
            return f"Error analyzing the video: {str(e)}", "An error occurred during analysis."
    
    def analyze_segment(self, query: str, start_time: float, end_time: float, segment_type: str) -> QueryVisionLLMResponseSegment:
        """
        Analyze a specific segment of the video.
        
        Args:
            query: The question to answer
            start_time: Start time of the segment in seconds
            end_time: End time of the segment in seconds
            segment_type: Description of the segment type (for logging and prompts)
            
        Returns:
            Tuple of (answer, explanation)
        """
        logger.log_info(f"Analyzing {segment_type} for query: {query} ({start_time:.2f}s to {end_time:.2f}s)")
        
        # Extract frames from the specified segment
        segment_duration = end_time - start_time
        num_frames = min(15, int(segment_duration * 2))
        frames, _ = resource_manager.extract_frames_between(
            num_frames=num_frames,
            start_time=start_time,
            end_time=end_time
        )
        
        if not frames:
            logger.log_error(f"Failed to extract frames for {segment_type} analysis")
            return f"Could not analyze the {segment_type}.", "No frames were extracted."
        
        # Create the prompt
        prompt = f"These frames show a specific segment of the video from {start_time:.2f}s to {end_time:.2f}s that is identified as the {segment_type} for answering: {query}"
        prompt += "\nFirst descript what you see in the video referring to the question. Then focus your analysis on this segment and provide both a direct answer and a detailed explanation of your reasoning to the question. If you think the answer is not clear, set need_zoom_in_to_single_image_of_video to True so that we can zoom in on the most important parts of the video and try again."
        #prompt += self.hint_prompt
        
        # Query the vision model
        request = VisionModelRequest(prompt, frames, response_class=QueryVisionLLMResponseSegment)
        try:
            response = query_vision_llm(request, model=self.model, display=self.display, system_prompt=self.get_system_prompt())
            logger.log_info(f"{segment_type.capitalize()} analysis - Answer: {response.answer}")
            logger.log_info(f"{segment_type.capitalize()} analysis - Explanation: {response.explanation[:100]}...")
            logger.log_info(f"{segment_type.capitalize()} analysis - Need zoom in to single image of video: {response.need_zoom_in_to_single_image_of_video}")
            return response
        except Exception as e:
            logger.log_error(f"Error in {segment_type} analysis: {str(e)}")
            return f"Error analyzing the {segment_type}: {str(e)}", "An error occurred during analysis."
    
    def analyze_key_image(self, query: str, image_time: float) -> QueryVisionLLMResponseKeyImage:
        """
        Analyze a single key image from the video in high detail.
        
        Args:
            query: The question to answer
            image_time: Timestamp of the key image in seconds
            
        Returns:
            Tuple of (answer, explanation)
        """
        logger.log_info(f"Analyzing key image at {image_time:.2f}s for query: {query}")
        
        try:
            # Extract the single frame at the specified timestamp
            frame, _ = resource_manager.get_frame_at_time(image_time)
            
            if frame is None or frame.size == 0:
                logger.log_error(f"Failed to extract key image at time {image_time:.2f}s")
                return "Could not analyze the key image.", "No image was extracted."
            
            # Create the prompt
            prompt = f"This image is identified as the most important moment in the video for answering the question: {query}"
            prompt += "\nAnalyze this image in high detail, paying attention to expressions, objects, and visual elements that might be relevant."
            prompt += "\nFirst descript what you see in the image in forms of WHO does WHAT with intention of WHAT. Then provide both a direct answer based solely on this key image and a detailed explanation of your reasoning. Give a confidence string as one of the following: 'high', 'medium', 'low' of whether the information is sufficient to answer the question."
            #prompt += self.hint_prompt
            
            # Create a VisionModelRequest with high detail
            request = VisionModelRequest(
                query=prompt, 
                images=[frame], 
                high_detail=True,
                response_class=QueryVisionLLMResponseKeyImage
            )
            
            # Query the vision model
            response = query_vision_llm(request, model=self.model, display=self.display, system_prompt=self.get_system_prompt())
            logger.log_info(f"Key image analysis - Answer: {response.answer}")
            logger.log_info(f"Key image analysis - Explanation: {response.explanation[:100]}...")
            logger.log_info(f"Key image analysis - Confidence: {response.confidence}")
            return response
        except Exception as e:
            logger.log_error(f"Error in key image analysis: {str(e)}")
            return f"Error analyzing the key image: {str(e)}", "An error occurred during analysis."
    
    def generate_hint_prompt(self, query: str, video_info: YoutubeVideoInfo) -> str:
        if not video_info.is_valid():
            return ""
        prompt = f"Please proces the following information about the youtube video. First, organise the information and try describe what the video is expected to be about. Next, Generate a hint prompt to help the LLM to answer the question: {query}"
        prompt += f"\nVideo information: {video_info.to_prompt()}"
        request = VisionModelRequest(prompt, [], response_class=QueryVisionLLMResponseHint)
        response = query_vision_llm(request, model=self.model, display=self.display, system_prompt=self.get_system_prompt())
        return response
    
    def get_answer(self, row: ParquetFileRow) -> str:
        """
        Get an answer to a question about a video.
        
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
            # Get video info
            video_info = get_youtube_video_info(row)
            hint_prompt = self.generate_hint_prompt(row.question, video_info)
            self.hint_prompt = hint_prompt.to_prompt()
            
            query = row.question
            
            # Step 1: Analyze the full video
            full_video_response = self.analyze_full_video(query)
            
            if not full_video_response.need_zoom_in_to_part_of_video:
                # If full video analysis is sufficient, return its answer
                logger.log_info("Full video analysis is sufficient, returning answer")
                return full_video_response.answer
            
            # Step 2: Identify the most important image/segment
            try:
                important_image_result = most_important_image_based_temporal_query(
                    resource_manager, 
                    query, 
                    system_prompt=self.get_system_prompt(), 
                    display=self.display, 
                    verbose=False
                )
            except Exception as img_error:
                logger.log_error(f"Error identifying important image: {str(img_error)}")
                # Fallback to full video answer if we can't identify key parts
                return full_video_response.answer
            
            # Step 3: Analyze the relevant segment
            relevant_segment_response = self.analyze_segment(
                query,
                important_image_result.start_time,
                important_image_result.end_time,
                "relevant segment"
            )
            
            # If segment analysis is sufficient, return its answer
            if not relevant_segment_response.need_zoom_in_to_single_image_of_video:
                logger.log_info("Relevant segment analysis is sufficient, returning answer")
                return relevant_segment_response.answer
            
            # Step 4 (optional): Analyze extended segment if not the last scene
            extended_segment_response = None
            if not important_image_result.is_last_scene:
                extended_segment_response = self.analyze_segment(
                    query,
                    important_image_result.start_time,
                    important_image_result.next_scene_end_time,
                    "extended segment"
                )
                
                # If extended segment gives better answer, use it
                if not extended_segment_response.need_zoom_in_to_single_image_of_video:
                    logger.log_info("Extended segment analysis is sufficient, returning answer")
                    return extended_segment_response.answer
            
            # Step 5: Analyze the key image in high detail
            key_image_response = self.analyze_key_image(
                query,
                important_image_result.image_time
            )
            
            # Step 6: Synthesize final answer based on all analyses
            # Construct a detailed prompt for the synthesizer
            synthesis_prompt = f"""
I have analyzed a video to answer the question: "{query}"

My analysis included several progressive steps:

1. FULL VIDEO ANALYSIS:
Video description: {full_video_response.description}
Answer: {full_video_response.answer}
Explanation: {full_video_response.explanation}

2. RELEVANT SEGMENT ANALYSIS:
Segment description: {relevant_segment_response.description}
Answer: {relevant_segment_response.answer}
Explanation: {relevant_segment_response.explanation}
"""

            # Add extended segment analysis if available
            if extended_segment_response:
                synthesis_prompt += f"""
3. EXTENDED SEGMENT ANALYSIS:
Segment description: {extended_segment_response.description}
Answer: {extended_segment_response.answer}
Explanation: {extended_segment_response.explanation}

4. KEY IMAGE ANALYSIS:
"""
            else:
                synthesis_prompt += f"""
3. KEY IMAGE ANALYSIS:
"""
                
            synthesis_prompt += f"""
Image description: {key_image_response.description}
Answer: {key_image_response.answer}
Explanation: {key_image_response.explanation}
Confidence: {key_image_response.confidence}

Based on all these analyses, please provide a comprehensive final answer to the original question.
Synthesize the information from each layer of analysis, giving more weight to the analyses of relevant parts and the key image if appropriate.
Provide a concise final answer that best addresses the question.
"""
            
            # Get final synthesized answer
            request = VisionModelRequest(
                query=synthesis_prompt,
                images=[],
                response_class=QueryVisionLLMResponseWithExplanation
            )
            
            final_response = query_vision_llm(
                request, 
                model=self.model,
                display=False, 
                system_prompt=self.get_system_prompt()
            )
            
            logger.log_info(f"Final synthesized answer: {final_response.answer}")
            
            # Return the final answer
            return f"Answer: {final_response.answer}. Explanation: {final_response.explanation}" 
            
        except Exception as e:
            logger.log_error(f"Error in L3Agent.get_answer: {str(e)}")
            return f"Error analyzing the video: {str(e)}"
    
    def get_agent_name(self) -> str:
        """
        Get the name of this agent.
        
        Returns:
            The name of the agent
        """
        return f"L3Agent_{self.model}"
    
    def get_system_prompt(self) -> str:
        """
        Get the system prompt for the agent.
        
        Returns:
            The system prompt string
        """
        return DEFAULT_SYSTEM_PROMPT + self.hint_prompt

# Export the L3Agent class
__all__ = ["L3Agent"] 