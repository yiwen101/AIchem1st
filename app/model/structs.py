from dataclasses import dataclass
from typing import List, Type
import cv2
import base64

import numpy as np
from pydantic import BaseModel

@dataclass
class ParquetFileRow:
    qid: str
    video_id: str
    question_type: str
    capability: str
    question: str
    duration: str
    question_prompt: str
    answer: str
    youtube_url: str

    def is_mcq(self):
        return self.question_type == "Multiple-choice Question with a Single Correct Answer"


@dataclass
class YoutubeVideoInfo:
    title: str
    video_length: str
    description: str
    transcript: str

    def to_prompt(self):
        if self.title == "":
            return ""
        return f"\n Here are some information about the video: \n Title: {self.title} \n Description: {self.description} \n Video Length: {self.video_length} \n Transcript: {self.transcript} \n"

    def is_valid(self):
        return self.title != ""
   

@dataclass
class AttemptAnswerResponse:
    can_answer: bool
    answer: str
    reasoning: str

@dataclass
class ToolCall:
    tool_name: str
    parameters: dict


@dataclass
class VisionModelRequestImage:
    base64_image: str
    high_detail: bool = False

    def __init__(self, image: np.ndarray, high_detail: bool = False):
        success, buffer = cv2.imencode('.jpg', image)
        if not success:
            raise ValueError("Failed to encode image")
        self.base64_image = base64.b64encode(buffer).decode('utf-8')
        self.high_detail = high_detail

    def to_request_json_object(self):
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{self.base64_image}",
                "detail": "high" if self.high_detail else "low"
            }
        }

class QueryVisionLLMResponse(BaseModel):
    answer: str

class QueryVisionLLMResponseWithExplanation(BaseModel):
    answer: str
    explanation: str

@dataclass
class VisionModelRequest:
    images: List[VisionModelRequestImage]
    query: str    
    response_class: Type[BaseModel]
    
    def __init__(self, query: str, images: List[np.ndarray], high_detail: bool = False, response_class: Type[BaseModel] = QueryVisionLLMResponseWithExplanation):
        """
        Initialize a vision model request.
        
        Args:
            query: The text query to ask
            images: A list of numpy array images (optional)
            image: A single numpy array image (optional)
            high_detail: Whether to use high detail mode for all images
            high_details: A list of high detail flags matching the images list
        """
        self.images = [VisionModelRequestImage(img, high_detail) for img in images]
        self.query = query
        self.response_class = response_class
    
    def to_json_array(self):
        """
        Create a content array for OpenAI's API.
        
        Returns:
            An array of content objects suitable for OpenAI's API
        """
        content_array = [{"type": "text", "text": self.query}]
        for image in self.images:
            content_array.append(image.to_request_json_object())
        return content_array


@dataclass
class QARecord:
    question: str
    answer: str
    reasoning: str
