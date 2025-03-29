"""
OpenAI LLM utilities for generating responses from various models.
"""

import os
import sys
import json
import base64
from typing import Any, Dict, Optional, Union, List
import numpy as np
import cv2
from openai import OpenAI

from app.common.monitor import logger

openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    logger.log_error("OPENAI_API_KEY environment variable not found.")
    logger.log_error("Please set it in your .env file or export it to your environment.")
    sys.exit(1)

client = OpenAI(
    api_key=openai_api_key,
)

def query_vision_llm(image: Union[str, np.ndarray], query: str, model: str = "gpt-4o") -> str:
    """
    Query the OpenAI vision model with an image and text prompt.
    
    Args:
        image: Either a path to an image file (str) or a numpy array containing the image
        query: Text prompt to send along with the image
        model: OpenAI model to use, defaults to gpt-4o
        
    Returns:
        The response text from the model
    """
    if isinstance(image, str):
        logger.log_info(f"Querying {model} vision model with image file: {image}")
        # Read and encode the image from file
        with open(image, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
    else:
        logger.log_info(f"Querying {model} vision model with provided image array")
        # Convert numpy array to base64
        success, buffer = cv2.imencode('.jpg', image)
        if not success:
            raise ValueError("Failed to encode image")
        base64_image = base64.b64encode(buffer).decode('utf-8')
    
    logger.log_llm_prompt(query)
    
    # Create the message with content including the image
    content = [
        {
            "type": "text",
            "text": query
        },
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
        }
    ]
    
    # Make the API call
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        max_tokens=300
    )
    
    response_text = response.choices[0].message.content
    logger.log_llm_response(response_text)
    
    return response_text 