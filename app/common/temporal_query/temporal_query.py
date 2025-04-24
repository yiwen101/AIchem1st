from app.common.utils.plot import PlotResponse, plot_video
from app.common.utils.scene import get_scene_seperated_frames
from app.model.structs import VisionModelRequest
from app.common.resource_manager.resource_manager import ResourceManager
from app.common.llm.openai import query_vision_llm
from typing import List, Tuple, Literal
from pydantic import BaseModel
import numpy as np
import os
import json
from app.common.monitor.logger import logger

class TemporalResponse(BaseModel):
    start_image_index: int
    end_image_index: int
    reasoning: str

class QueryAnalysis(BaseModel):
    query_type: Literal["full_video", "relevant_part", "single_image"]
    reasoning: str

class MostImportantImageLLMResponse(BaseModel):
    most_important_image_index: int
    reasoning: str

class MostImportantImageResponseVerbose(BaseModel):
    most_important_image_index: int
    reasoning: str
    video_description: str
    image_description: List[str]

class MostImportantImageResponse:
    start_time: float
    end_time: float
    image_time: float
    is_last_scene: bool
    next_scene_end_time: float

    def __init__(self, start_time: float, end_time: float, image_time: float, is_last_scene: bool, next_scene_end_time: float):
        self.start_time = start_time
        self.end_time = end_time
        self.image_time = image_time
        self.is_last_scene = is_last_scene
        self.next_scene_end_time = next_scene_end_time

def get_or_create_video_plot(resource_manager: ResourceManager, method_name: str = "scene_based_sampling", system_prompt: str = "", display: bool = False) -> PlotResponse:
    """
    Get or create a video plot JSON file with scene information and image descriptions.
    
    Args:
        resource_manager: The resource manager with an active video
        method_name: Name of the method/folder to store the plot
        system_prompt: System prompt for the LLM
        display: Whether to display images when querying LLM
        
    Returns:
        Dictionary with video plot information
    """
    # Get active video metadata
    _, metadata = resource_manager.get_active_video()
    video_id = metadata["video_name"]
    
    # Define the persistence path
    persistence_dir = f"persistence/video_plot/{method_name}"
    os.makedirs(persistence_dir, exist_ok=True)
    json_path = f"{persistence_dir}/{video_id}.json"
    
    # Check if JSON already exists
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                video_plot_json = json.load(f)
                video_plot = PlotResponse.from_json(video_plot_json)
                logger.log_info(f"Loaded existing video plot for {video_id}")
                return video_plot
        except Exception as e:
            logger.log_error(f"Error loading video plot: {str(e)}")
            # Continue to create a new one

    plot_response = plot_video(resource_manager, system_prompt, display, method_name = method_name)
    logger.log_info(f"temporal query, line 73, video plot created with {len(plot_response.captions)} captions")
    # Save to JSON file
    try:
        with open(json_path, 'w') as f:
            json.dump(plot_response.to_json(), f, indent=2)
        logger.log_info(f"Saved video plot to {json_path}")
        logger.log_info(f"temporal query, line 79, plot has video description: '{plot_response.video_description[:50]}...'")
    except Exception as e:
        logger.log_error(f"Error saving video plot: {str(e)}")
    
    return plot_response

def caption_based_analyze_temporal_query(resource_manager: ResourceManager, query: str, method_name: str = "scene_based_sampling", system_prompt: str = "", display: bool = False) -> Tuple[float, float]:
    """
    Analyze a query to determine the best temporal approach and return relevant timestamps.
    
    Args:
        resource_manager: The resource manager with an active video
        query: The query to analyze
        method_name: Name of the method/folder to store the plot
        system_prompt: System prompt for the LLM
        display: Whether to display images when querying LLM
        
    Returns:
        Tuple of (start_time, end_time) in seconds
    """
    # Get or create video plot
    video_plot = get_or_create_video_plot(resource_manager, method_name, system_prompt, display)
    
    # Create a prompt for the LLM to analyze the query
    prompt = f"""
Analyze what kind of temporal information is needed to answer the query "{query}" for the video:
1. "full_video" - The entire video needs to be considered 
2. "relevant_part" - Only a specific part of the video is needed
3. "single_image" - A single detailed image would be best

Video description: {video_plot.to_prompt()}
"""
    
    # Ask LLM to analyze the query
    request = VisionModelRequest(
        query=prompt,
        images=[],
        response_class=QueryAnalysis,
    )
    if system_prompt != "":
        response = query_vision_llm(request, display=display, system_prompt=system_prompt)
    else:
        response = query_vision_llm(request, display=display)
    
    logger.log_info(f"Query analysis result: {response}")
    
    video_duration = video_plot.video_duration
    
    # Handle different query types
    if response.query_type == "full_video":
        # Return full video duration
        return 0.0, video_duration
    
    elif response.query_type == "relevant_part":
        prompt = f"""
You are tasked to find the segment of video for answering the query. A shorter segment leads to more focused and high precision answer, but it might lack information.
I will provide you with a video description and a list of images descriptions in chronological order from the video. Please study the description carefully, and find the first and last image index that is most relevant to the query, so that I can extract the video segment between them.
The query is: "{query}"
The video description is: {video_plot.to_prompt()}
        """
        request = VisionModelRequest(
            query=prompt,
            images=[],
            response_class=TemporalResponse,
        )
        if system_prompt != "":
            response = query_vision_llm(request, display=display, system_prompt=system_prompt)
        else:
            response = query_vision_llm(request, display=display)
        
        start_time = video_plot.captions[response.start_image_index-1].image_time
        end_time = video_plot.captions[response.end_image_index-1].image_time
        return start_time, end_time
    
    elif response.query_type == "single_image":
        # This is not implemented yet
        raise NotImplementedError("Single image detailed analysis not implemented yet")
    
    # Default fallback - return full video
    return 0.0, video_duration

def most_important_image_based_temporal_query(resource_manager: ResourceManager, query: str, system_prompt: str = "", display: bool = False, verbose: bool = False) -> MostImportantImageResponse:
    """
    Analyze a query by finding the single most important image, then returning the scene time range it belongs to.
    
    Args:
        resource_manager: The resource manager with an active video
        query: The query to analyze
        method_name: Name of the method/folder to store the plot
        system_prompt: System prompt for the LLM
        display: Whether to display images when querying LLM
        
    Returns:
        Tuple of (start_time, end_time) in seconds for the scene containing the most important image
    """  
    # Extract frames and scene info
    logger.log_info("Getting scene seperated frames")
    frames, scene_info, timestamps = get_scene_seperated_frames(resource_manager)
    use_fallback = len(frames) >= 20
    if use_fallback:
        logger.log_info("Extracting 20 frames")
        frames, timestamps = resource_manager.extract_frames_between(20, save_frames=False)
        logger.log_info(f"Extracted {len(frames)} frames")
    if verbose:
        important_image_prompt = f"""
        The video contains the following images in chronological order. Based on the query: "{query}"
        Please first describe the video overall. Then describe each of the {len(frames)} images.
        Next, most importantly, please identify the SINGLE most important image from the sequence above that best answers or addresses the query.
        Return the image number (1-indexed) that is most relevant, and a brief reasoning for your choice.
        """
        request = VisionModelRequest(
            query=important_image_prompt,
            images=frames,
            response_class=MostImportantImageResponseVerbose,
        )
    else:
        important_image_prompt = f"""
    The video contains the following images in chronological order. Based on the query: "{query}"
    Please identify the SINGLE most important image from the sequence above that best answers or addresses the query.
    Return only the image number (1-indexed) that is most relevant, and a brief reasoning for your choice.
    """
        
        request = VisionModelRequest(
            query=important_image_prompt,
            images=frames,
            response_class=MostImportantImageLLMResponse,
        )
    
    response = query_vision_llm(request, display=display, system_prompt=system_prompt)
    most_important_image_index = response.most_important_image_index
    image_time = timestamps[most_important_image_index-1]
    logger.log_info(f"Most important image response: image_index={most_important_image_index}, reasoning={response.reasoning[:100]}...")
    if use_fallback:
        prev_index = max(0, most_important_image_index - 1)
        next_index = min(len(timestamps) - 1, most_important_image_index + 1)
        start_time = timestamps[prev_index]
        end_time = timestamps[next_index]

        if most_important_image_index == len(timestamps):
            return MostImportantImageResponse(start_time, end_time, image_time, True, -1)
        next_scene_index = min(most_important_image_index + 3, len(scene_info) - 1)
        next_scene_end_time = timestamps[next_scene_index]
        return MostImportantImageResponse(start_time, end_time, image_time, False, next_scene_end_time)
        
    else:
        scene_index = (most_important_image_index - 1) // 3
        start_time = scene_info[scene_index]['start_time']
        end_time = scene_info[scene_index]['end_time']
        if scene_index == len(scene_info) - 1:
            return MostImportantImageResponse(start_time, end_time, image_time, True, -1)
        next_scene_end_time = scene_info[scene_index+1]['end_time']
        return MostImportantImageResponse(start_time, end_time, image_time, False, next_scene_end_time)