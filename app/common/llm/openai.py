"""
OpenAI LLM utilities for generating responses from various models.
"""

import os
import sys
import json
import base64
from typing import Any, Dict, Optional, Type, Union, List
import numpy as np
import cv2
from openai import OpenAI
from pydantic import BaseModel

from app.common.monitor import logger
from app.model.structs import VisionModelRequest

openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    logger.log_error("OPENAI_API_KEY environment variable not found.")
    logger.log_error("Please set it in your .env file or export it to your environment.")
    sys.exit(1)

client = OpenAI(
    api_key=openai_api_key,
)

def query_llm_text(request: str, model: str = "gpt-4o-mini") -> str:
    """
    Query the OpenAI LLM with a text prompt.
    """
    logger.log_llm_prompt(request)
    
    resp =  client.chat.completions.create(model=model, messages=[{"role": "user", "content": request}])
    logger.log_llm_response(resp.choices[0].message.content)
    print(resp.choices[0].message.content)
    return resp.choices[0].message.content

#https://platform.openai.com/docs/guides/structured-outputs?api-mode=chat
class QueryVisionLLMResponse(BaseModel):
    answer: str

class QueryVisionLLMResponseWithExplanation(BaseModel):
    answer: str
    explanation: str

def query_llm(model: str, messages: List[Dict[str, Any]]) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=2000,
    )
    return response.choices[0].message.content

def query_llm_structured(model: str, messages: List[Dict[str, Any]], response_class: Type[BaseModel]) -> BaseModel:
    response = client.beta.chat.completions.parse(
        model=model,
        messages=messages,
        max_tokens=2000,
        response_format=response_class,
    )
    return response.choices[0].message.parsed

def query_vision_llm(request: VisionModelRequest, model: str = "gpt-4o-mini") -> QueryVisionLLMResponseWithExplanation:
    """
    Query the OpenAI vision model with an image and text prompt.
    
    Args:
        request: A VisionModelRequest object containing query and images
        model: OpenAI model to use, defaults to gpt-4o-mini
        
    Returns:
        The response text from the model
    """    
    logger.log_llm_prompt(request.query)
    
    # Prepare the content array directly
    content_array = [{"type": "text", "text": request.query}]
    for image in request.images:
        content_array.append(image.to_request_json_object())
    
    messages=[{"role": "user", "content": content_array}]
    
    # Make the API call with properly formatted message
    response = query_llm_structured(model, messages, QueryVisionLLMResponseWithExplanation if request.require_explanation else QueryVisionLLMResponse)
    
    logger.log_llm_response(response)
    if not request.require_explanation:
        return QueryVisionLLMResponseWithExplanation(answer=response.answer, explanation="")
    else:
        return response

def query_vision_llm_single_image(image: np.ndarray, query: str, model: str = "gpt-4o-mini") -> str:
    """
    Query the OpenAI vision model with a single image and text prompt.
    
    Args:
        image: A numpy array containing the image
        query: Text prompt to send along with the image
        model: OpenAI model to use, defaults to gpt-4o-mini
        
    Returns:
        The response text from the model
    """
    return query_vision_llm(VisionModelRequest(query, [image]), model)
    