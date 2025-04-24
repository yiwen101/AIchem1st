"""
OpenAI LLM utilities for generating responses from various models.
"""

import os
import sys
import json
import base64
import time
from typing import Any, Dict, Optional, Type, Union, List
import numpy as np
import cv2
from openai import OpenAI
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv()

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




def query_llm(model: str, messages: List[Dict[str, Any]]) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=2000,
    )
    return response.choices[0].message.content

def query_llm_structured(model: str, messages: List[Dict[str, Any]], response_class: Type[BaseModel]) -> BaseModel:
    for _ in range(3):
        try:
            response = client.beta.chat.completions.parse(
                model=model,
                messages=messages,
                max_tokens=2000,
                response_format=response_class,
            )
            return response.choices[0].message.parsed
        except Exception as e:
            logger.log_error(f"Error querying LLM structured: {e}")
            time.sleep(1)
    raise Exception("Failed to query LLM structured")

def single_query_llm_structured(model: str, query: str, response_class: Type[BaseModel]) -> BaseModel:
    messages=[{"role": "user", "content": query}]
    return query_llm_structured(model, messages, response_class)

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant that can answer questions about video based on frames extracted from the video."

def query_vision_llm(request: VisionModelRequest, model: str = "gpt-4o-mini", display: bool = False, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> Any:
    """
    Query the OpenAI vision model with an image and text prompt.
    
    Args:
        request: A VisionModelRequest object containing query and images
        model: OpenAI model to use, defaults to gpt-4o-mini
        display: If True, display the images in a grid along with prompt and response
        
    Returns:
        The response text from the model
    """    
    logger.log_llm_prompt(request.query)
    # Prepare the content array directly
    content_array = [{"type": "text", "text": request.query}]
    for image in request.images:
        content_array.append(image.to_request_json_object())
    
    messages = []
    if system_prompt != "":
        messages.append({"role": "system", "content": system_prompt})
    
    messages.append({"role": "user", "content": content_array})
    
    # Make the API call with properly formatted message
    response = query_llm_structured(model, messages, request.response_class)
    logger.log_llm_response(response)
    
    # Display images and response if requested
    if display:
        import matplotlib.pyplot as plt
        from math import ceil, sqrt
        
        # Calculate grid size
        n = len(request.images)
        response_str = str(response)
        plot_text = f"Prompt: {system_prompt}\n{request.query}\nResponse: {response_str}"
        if n == 0:
            # show the text
            plt.figure(figsize=(4, 4))
            plt.text(0.5, 0.5, plot_text, ha='center', va='center', fontsize=12)
            plt.axis('off')
            plt.show()
            return response
            
        cols = ceil(sqrt(n))
        rows = ceil(n / cols)
        
        plt.figure(figsize=(4*cols, 4*rows + 2))
        
        # Plot images
        for i, img_req in enumerate(request.images):
            img_data = base64.b64decode(img_req.base64_image)
            img_array = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            plt.subplot(rows + 1, cols, i + 1)
            plt.imshow(img)
            plt.title(f"Image {i+1}")
            plt.axis('off')
        
        # Add prompt and response
        plt.figtext(0.5, 0.02, plot_text,  
                   wrap=True, horizontalalignment='center', fontsize=12)
        
        plt.tight_layout()
        plt.show()
    
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
    