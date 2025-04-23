from pydantic import BaseModel
from typing import List
import numpy as np

from app.common.llm.openai import query_vision_llm
from app.common.resource_manager.resource_manager import ResourceManager
from app.common.utils.scene import get_scene_seperated_frames
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

class Caption:
    image_index: int
    image_time: float
    caption: str
    
    def __init__(self, image_index, image_time, caption):
        self.image_index = image_index
        self.image_time = image_time
        self.caption = caption
    
    def __str__(self):
        return f"Caption(idx={self.image_index}, time={self.image_time:.2f}s, caption='{self.caption[:30]}...')"
    
    def to_dict(self):
        return {
            "image_index": self.image_index,
            "image_time": self.image_time,
            "caption": self.caption
        }

class PlotResponse:
    video_description: str
    video_duration: float
    captions: List[Caption]

    def __init__(self, video_description: VideoDescription, times: List[float]):
        if len(times) != len(video_description.image_captions):
            raise ValueError("The length of times and image_captions must be the same")
        self.video_description = video_description.video_description
        self.captions = [Caption(i+1, times[i], video_description.image_captions[i]) for i in range(len(times))]
        self.video_duration = times[-1]
    
    def __str__(self):
        return f"PlotResponse(video_description='{self.video_description}', duration={self.video_duration}, captions={len(self.captions)})"
    
    def to_json(self):
        return {
            "video_description": self.video_description,
            "video_duration": self.video_duration,
            "captions": [caption.to_dict() for caption in self.captions]
        }
    
    @classmethod
    def from_json(cls, json_data: dict):
        plot_response = cls.__new__(cls)
        plot_response.video_description = json_data["video_description"]
        plot_response.video_duration = json_data["video_duration"]
        plot_response.captions = [Caption(
            caption["image_index"], 
            caption["image_time"], 
            caption["caption"]
        ) for caption in json_data["captions"]]
        return plot_response
    
    def to_prompt(self):
        prompt = f"The video is about {self.video_description}. The video duration is {self.video_duration} seconds.\n"
        for i, caption in enumerate(self.captions):
            prompt += f"Image {i+1}: {caption.caption}\n"
        return prompt




def plot_video(resource_manager: ResourceManager, system_prompt: str, display: bool = False) -> PlotResponse:
    """
    Ask the user to plot the video frames.
    """
    frames, scene_info, times = get_scene_seperated_frames(resource_manager)
    
    logger.log_info(f"Scene info: {scene_info}")
    prompt = f"The images are extracted from a video in chronological order. Please generate caption for each of the {len(frames)} images, and then describe what the whole video is about. Focus on factual information, not evaluation. Make inference about what happen in between images by contrasting adjacent images."
    request = VisionModelRequest(
        query=prompt,
        images=frames,
        response_class=VideoDescription,
    )
    response = query_vision_llm(request, display=display, system_prompt=system_prompt)
    logger.log_info(f"plot, line 86, {response}")
    plot_response = PlotResponse(response, times)
    logger.log_info(f"plot, line 88, {plot_response}")
    return plot_response
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

