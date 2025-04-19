"""
Object detection tools for video analysis using Hugging Face models.

This module provides tools for detecting objects in video frames using pre-trained transformer models.
"""

import os
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from typing import Dict, Any, List, Optional
import matplotlib.pyplot as plt
from transformers import AutoImageProcessor, AutoModelForObjectDetection

from app.tools.toolImpl.base_tool import BaseTool, ToolParameter, ToolParameterType
from app.tools.tool_manager import register_tool
from app.common.resource_manager import resource_manager
from app.common.monitor import logger

# Cache for models to avoid reloading
model_cache = {}

def load_detection_model(model_name="facebook/detr-resnet-50"):
    """
    Load and cache the object detection model from Hugging Face.
    
    Args:
        model_name: Name of the model on Hugging Face
        
    Returns:
        Tuple of (processor, model)
    """
    # Check if model is already cached
    if model_name in model_cache:
        logger.log_info(f"Using cached model: {model_name}")
        return model_cache[model_name]
    
    logger.log_info(f"Loading model: {model_name}")
    try:
        # Load model and processor
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModelForObjectDetection.from_pretrained(model_name)
        
        # Cache the loaded model
        model_cache[model_name] = (processor, model)
        logger.log_info(f"Successfully loaded model: {model_name}")
        return processor, model
    except Exception as e:
        logger.log_error(f"Error loading model {model_name}: {str(e)}")
        raise

@register_tool
class ObjectDetectionTool(BaseTool):
    """Tool for detecting objects in video frames using transformer models."""
    
    name = "object_detection"
    description = "Detect objects in a video frame at specified time using transformer-based models."
    parameters = [
        ToolParameter(
            name="time_seconds",
            type=ToolParameterType.FLOAT,
            description="Time in seconds (e.g., 70.45 for 70 seconds and 45 milliseconds)",
            required=True
        ),
        ToolParameter(
            name="confidence_threshold",
            type=ToolParameterType.FLOAT,
            description="Minimum confidence for detection",
            required=False,
            default=0.5
        ),
        ToolParameter(
            name="model_name",
            type=ToolParameterType.STRING,
            description="Hugging Face model to use for detection",
            required=False,
            default="facebook/detr-resnet-50"
        )
    ]
    
    @classmethod
    def execute(cls, time_seconds: float, confidence_threshold: float = 0.5, 
               model_name: str = "facebook/detr-resnet-50") -> Dict[str, Any]:
        """
        Detect objects in a video frame at specified time.
        
        Args:
            time_seconds: Time in seconds (e.g., 70.45 for 70 seconds and 45 milliseconds)
            confidence_threshold: Minimum confidence for detection, default is 0.5
            model_name: Hugging Face model name, default is "facebook/detr-resnet-50"
            
        Returns:
            Dictionary with detected objects
        """
        # Get frame at specified time
        frame, frame_index = resource_manager.get_frame_at_time(time_seconds)
        
        # Convert OpenCV frame (BGR) to RGB for PIL
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        
        # Load model
        try:
            processor, model = load_detection_model(model_name)
        except Exception as e:
            logger.log_error(f"Failed to load model: {str(e)}")
            return {
                "error": f"Failed to load model: {str(e)}",
                "time": time_seconds
            }
        
        # Process image and get predictions
        logger.log_info(f"Processing frame at time {time_seconds} seconds")
        inputs = processor(images=pil_image, return_tensors="pt")
        outputs = model(**inputs)
        
        # Convert outputs to detections
        target_sizes = torch.tensor([pil_image.size[::-1]])
        results = processor.post_process_object_detection(
            outputs, 
            target_sizes=target_sizes, 
            threshold=confidence_threshold
        )[0]
        
        # Format results
        detected_objects = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            # Convert tensor values to Python types
            score_val = score.item()
            label_id = label.item()
            box_vals = [int(b) for b in box.tolist()]
            
            # Get label name from model's id2label mapping
            label_name = model.config.id2label[label_id]
            
            x, y, x2, y2 = box_vals
            width = x2 - x
            height = y2 - y
            
            detected_objects.append({
                "label": label_name,
                "confidence": float(score_val),
                "box": {
                    "x": int(x),
                    "y": int(y),
                    "width": int(width),
                    "height": int(height)
                },
                "center": {
                    "x": int(x + width/2),
                    "y": int(y + height/2)
                }
            })
        
        logger.log_info(f"Detected {len(detected_objects)} objects")
        
        # Create annotated image
        annotated_image = pil_image.copy()
        draw = ImageDraw.Draw(annotated_image)
        
        for obj in detected_objects:
            x = obj["box"]["x"]
            y = obj["box"]["y"]
            width = obj["box"]["width"]
            height = obj["box"]["height"]
            
            # Draw rectangle
            draw.rectangle(
                [(x, y), (x + width, y + height)],
                outline="red",
                width=3
            )
            
            # Draw label
            label = f"{obj['label']}: {obj['confidence']:.2f}"
            draw.text((x, y - 10), label, fill="red")
        
        # Convert back to OpenCV format for saving
        annotated_frame = np.array(annotated_image)
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
        
        # Save annotated image
        output_path = resource_manager.save_image(
            annotated_frame, 
            time_seconds, 
            "object_detection"
        )
        
        return {
            "detected_objects": detected_objects,
            "count": len(detected_objects),
            "time": time_seconds,
            "output_path": output_path
        } 