"""
Image captioning tools for video analysis.

This module provides tools for generating captions for images.
"""

import os
import cv2
import numpy as np
from typing import Dict, Any, List, Optional
import base64
import json
import requests

from app.tools.tool_registry import register_tool

@register_tool("image_captioning")
def image_captioning(
    image_path: str,
    max_tokens: int = 50,
    temperature: float = 0.7,
    api_url: Optional[str] = None,
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate a caption for an image.
    
    Args:
        image_path: Path to the input image
        max_tokens: Maximum token length for generated caption
        temperature: Sampling temperature (higher = more creative)
        api_url: Optional custom API endpoint
        api_key: Optional API key for the service
        
    Returns:
        Dictionary with the generated caption and metadata
    """
    # Check if image exists
    if not os.path.exists(image_path):
        raise ValueError(f"Image file not found: {image_path}")
    
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Get API key from environment variable if not provided
    if not api_key:
        api_key = os.environ.get("IMAGE_CAPTION_API_KEY")
    
    # If API key is provided, use remote service
    if api_key:
        caption, model_used = _remote_captioning(image_path, api_key, api_url, max_tokens, temperature)
    else:
        # Fallback to local captioning if no API key
        caption, model_used = _local_captioning(image, max_tokens)
    
    # Extract image info
    height, width, channels = image.shape
    file_size = os.path.getsize(image_path)
    
    return {
        "image_path": image_path,
        "width": width,
        "height": height,
        "caption": caption,
        "model_used": model_used,
        "file_size": file_size
    }

def _local_captioning(image: np.ndarray, max_tokens: int) -> tuple[str, str]:
    """
    Generate a caption locally using a simple heuristic approach.
    
    Args:
        image: Image as numpy array
        max_tokens: Maximum token length
        
    Returns:
        Tuple of (caption, model_name)
    """
    # This is a very simple fallback that analyzes basic properties
    height, width, channels = image.shape
    
    # Extract some basic image properties
    avg_color = np.mean(image, axis=(0, 1))
    brightness = np.mean(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    edges = cv2.Canny(image, 100, 200)
    edge_density = np.count_nonzero(edges) / (height * width)
    
    # Very simple heuristic classification
    image_type = "photograph"
    if edge_density > 0.1:
        if brightness > 200:
            image_type = "diagram or drawing"
        else:
            image_type = "detailed image"
    
    # Color description
    color_desc = ""
    if avg_color[0] > 150:
        color_desc = "bluish"
    elif avg_color[1] > 150:
        color_desc = "greenish"
    elif avg_color[2] > 150:
        color_desc = "reddish"
    else:
        if brightness < 50:
            color_desc = "dark"
        elif brightness > 200:
            color_desc = "bright"
        else:
            color_desc = "medium-toned"
    
    # Create caption based on simple analysis
    resolution = "high" if width * height > 1000000 else "medium" if width * height > 400000 else "low"
    orientation = "portrait" if height > width else "landscape" if width > height else "square"
    
    caption = f"A {resolution}-resolution {orientation} {color_desc} {image_type}."
    
    # Truncate to max tokens (roughly characters / 4)
    if len(caption) > max_tokens * 4:
        caption = caption[:max_tokens * 4] + "..."
    
    return caption, "basic_image_analyzer"

def _remote_captioning(
    image_path: str, 
    api_key: str, 
    api_url: Optional[str],
    max_tokens: int,
    temperature: float
) -> tuple[str, str]:
    """
    Generate a caption using a remote API service.
    
    Args:
        image_path: Path to the image
        api_key: API key for the service
        api_url: API URL (if None, uses default)
        max_tokens: Maximum tokens for generation
        temperature: Sampling temperature
        
    Returns:
        Tuple of (caption, model_name)
    """
    # Default to a common API endpoint if none provided
    if not api_url:
        api_url = "https://api.example.com/image-captioning"
    
    # Read image and convert to base64
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
    
    # Prepare request
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    payload = {
        "image": encoded_image,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    
    # Send request
    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        
        # Parse response
        result = response.json()
        caption = result.get("caption", "No caption generated")
        model = result.get("model", "unknown_remote_model")
        
        return caption, model
    
    except requests.exceptions.RequestException as e:
        # Fallback to local captioning if API fails
        image = cv2.imread(image_path)
        caption, model = _local_captioning(image, max_tokens)
        return f"{caption} (API Error: {str(e)})", f"fallback_{model}" 