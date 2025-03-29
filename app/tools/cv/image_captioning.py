"""
Image captioning tool using OpenAI's vision models.

This module provides functionality to generate detailed captions for images
using OpenAI's multimodal models.
"""

import os
import base64
from typing import Dict, Any, List, Optional
from PIL import Image
import io

from openai import OpenAI

from app.tools.base_tool import BaseTool, ToolParameter, ToolParameterType
from app.tools.tool_manager import register_cv_tool


# Cache for the OpenAI client
_openai_client = None

def get_openai_client():
    """
    Get or create an OpenAI client.
    
    Returns:
        OpenAI client
        
    Raises:
        ValueError: If OPENAI_API_KEY environment variable is not set
    """
    global _openai_client
    
    if _openai_client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        _openai_client = OpenAI(api_key=api_key)
    
    return _openai_client


@register_cv_tool
class CaptionImageTool(BaseTool):
    """Tool for generating detailed captions for images using vision models."""
    
    name = "caption_image"
    description = "Generate a detailed caption for an image using OpenAI's vision models. Useful for understanding image content."
    parameters = [
        ToolParameter(
            name="image_path",
            type=ToolParameterType.STRING,
            description="Path to the image file",
            required=True
        ),
        ToolParameter(
            name="model",
            type=ToolParameterType.STRING,
            description="OpenAI model to use (must support vision)",
            required=False,
            default="gpt-4o"
        ),
        ToolParameter(
            name="prompt",
            type=ToolParameterType.STRING,
            description="Text prompt to guide caption generation",
            required=False,
            default="Please provide a detailed description of this image."
        ),
        ToolParameter(
            name="max_tokens",
            type=ToolParameterType.INTEGER,
            description="Maximum tokens for the response",
            required=False,
            default=500
        )
    ]
    
    @classmethod
    def execute(cls,
                image_path: str, 
                model: str = "gpt-4o",
                prompt: str = "Please provide a detailed description of this image.",
                max_tokens: int = 500) -> Dict[str, Any]:
        """
        Generate a detailed caption for an image using OpenAI's vision models.
        
        Args:
            image_path: Path to the image file
            model: OpenAI model to use (must support vision)
            prompt: Text prompt to guide caption generation
            max_tokens: Maximum tokens for the response
            
        Returns:
            Dictionary with the image path and generated caption
            
        Raises:
            ValueError: If the image cannot be loaded or API call fails
        """
        # Load the image
        try:
            image = Image.open(image_path)
            
            # Convert image to base64
            buffered = io.BytesIO()
            image.save(buffered, format=image.format or "JPEG")
            base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
        except Exception as e:
            raise ValueError(f"Could not load image from {image_path}: {str(e)}")
        
        # Get OpenAI client
        client = get_openai_client()
        
        try:
            # Generate caption
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=max_tokens
            )
            
            # Extract the caption
            caption = response.choices[0].message.content
            
            return {
                "image_path": image_path,
                "caption": caption,
                "model_used": model
            }
        except Exception as e:
            raise ValueError(f"Error generating caption: {str(e)}")


@register_cv_tool
class BatchCaptionImagesTool(BaseTool):
    """Tool for generating captions for multiple images in batch."""
    
    name = "batch_caption_images"
    description = "Generate captions for multiple images in batch using OpenAI's vision models. Useful for analyzing multiple images at once."
    parameters = [
        ToolParameter(
            name="image_paths",
            type=ToolParameterType.ARRAY,
            description="List of paths to image files",
            required=True
        ),
        ToolParameter(
            name="model",
            type=ToolParameterType.STRING,
            description="OpenAI model to use (must support vision)",
            required=False,
            default="gpt-4o"
        ),
        ToolParameter(
            name="prompt",
            type=ToolParameterType.STRING,
            description="Text prompt to guide caption generation",
            required=False,
            default="Please provide a detailed description of this image."
        ),
        ToolParameter(
            name="max_tokens",
            type=ToolParameterType.INTEGER,
            description="Maximum tokens for each response",
            required=False,
            default=300
        )
    ]
    
    @classmethod
    def execute(cls,
                image_paths: List[str],
                model: str = "gpt-4o",
                prompt: str = "Please provide a detailed description of this image.",
                max_tokens: int = 300) -> Dict[str, Any]:
        """
        Generate captions for multiple images in batch.
        
        Args:
            image_paths: List of paths to image files
            model: OpenAI model to use (must support vision)
            prompt: Text prompt to guide caption generation
            max_tokens: Maximum tokens for each response
            
        Returns:
            Dictionary with results for each image
        """
        results = []
        
        for image_path in image_paths:
            try:
                caption_result = CaptionImageTool.execute(
                    image_path=image_path,
                    model=model,
                    prompt=prompt,
                    max_tokens=max_tokens
                )
                results.append(caption_result)
            except Exception as e:
                results.append({
                    "image_path": image_path,
                    "error": str(e),
                    "caption": None
                })
        
        return {
            "results": results,
            "total_images": len(image_paths),
            "successful_captions": sum(1 for r in results if "caption" in r and r["caption"] is not None)
        } 