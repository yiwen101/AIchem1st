from pydantic import BaseModel
from typing import List
import numpy as np

from app.common.llm.openai import query_vision_llm
from app.common.resource_manager.resource_manager import ResourceManager
from app.common.utils.scene import detect_scenes
from app.model.structs import VisionModelRequest
from app.common.monitor.logger import logger

'''
class SceneDescription(BaseModel):
    start_image_index: int
    end_image_index: int
    description: str

class PlotResponse(BaseModel):
    video_description: str
    scenes: List[SceneDescription]
    start_relevant_image_index: int
    end_relevant_image_index: int
'''

class VideoDescription(BaseModel):
    image_captions: List[str]
    video_description: str

   


def plot_video_frames(resource_manager: ResourceManager, system_prompt: str, display: bool = False):
    """
    Ask the user to plot the video frames.
    """
    frames, scene_info = detect_scenes(resource_manager)
    logger.log_info(f"Scene info: {scene_info}")
    prompt = f"The images are extracted from a video in chronological order. Please generate caption for each image, and finally describe what the whole video is about."
    request = VisionModelRequest(
        query=prompt,
        images=frames,
        response_class=VideoDescription,
    )
    response = query_vision_llm(request, display=display, system_prompt=system_prompt)
    prompt = f"The video is about {response.video_description}.\n"
    for i, caption in enumerate(response.image_captions):
        prompt += f"Image {i+1}: {caption}\n"
    prompt += "Please help me find the start and end image indexes for answer the query 'What's the baby's reaction when they see dad?'"
    request = VisionModelRequest(
        query=prompt,
        images=[]
    )
    response = query_vision_llm(request, display=display, system_prompt=system_prompt)
    return response

