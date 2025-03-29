"""
Object detection tools for video analysis.

This module provides tools for detecting objects in images or video frames.
"""

import os
import cv2
import numpy as np
from typing import Dict, Any, List, Optional

from app.tools.tool_registry import register_tool

@register_tool("object_detection")
def object_detection(
    image_path: str,
    confidence_threshold: float = 0.5,
    nms_threshold: float = 0.4,
    output_path: Optional[str] = None,
    model_path: Optional[str] = None,
    config_path: Optional[str] = None,
    classes_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Detect objects in an image using YOLOv3.
    
    Args:
        image_path: Path to the input image
        confidence_threshold: Minimum confidence for detection
        nms_threshold: Non-maximum suppression threshold
        output_path: Path to save the annotated image (optional)
        model_path: Path to custom model weights (optional)
        config_path: Path to custom model config (optional)
        classes_path: Path to custom classes file (optional)
        
    Returns:
        Dictionary with detected objects and their details
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    height, width = image.shape[:2]
    
    # Set up model paths
    if not model_path:
        model_path = os.path.join(os.path.dirname(__file__), "../data/models/yolov3.weights")
    if not config_path:
        config_path = os.path.join(os.path.dirname(__file__), "../data/models/yolov3.cfg")
    if not classes_path:
        classes_path = os.path.join(os.path.dirname(__file__), "../data/models/coco.names")
    
    # Load class names
    try:
        with open(classes_path, 'r') as f:
            classes = [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        # Fallback to COCO class names if file not found
        classes = [
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
            "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
            "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
            "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
            "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
            "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
            "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
            "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
            "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
        ]
    
    # Load the network
    try:
        net = cv2.dnn.readNetFromDarknet(config_path, model_path)
    except:
        raise ValueError(f"Could not load model from {model_path} with config {config_path}")
    
    # Get output layer names
    layer_names = net.getLayerNames()
    try:
        # OpenCV 4.x API
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    except:
        # Fallback for older OpenCV versions
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    
    # Create blob from image
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    
    # Forward pass
    net.setInput(blob)
    outputs = net.forward(output_layers)
    
    # Process outputs
    boxes = []
    confidences = []
    class_ids = []
    
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > confidence_threshold:
                # Scale bounding box coordinates back relative to image size
                box = detection[0:4] * np.array([width, height, width, height])
                centerX, centerY, w, h = box.astype('int')
                
                # Using center coordinates, calculate top-left corner
                x = int(centerX - (w / 2))
                y = int(centerY - (h / 2))
                
                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    # Apply non-maximum suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)
    
    # Prepare results
    detected_objects = []
    
    # Ensure indices is properly unwrapped (OpenCV 4.x vs older)
    if len(indices) > 0:
        try:
            # OpenCV 4.x
            indices = indices.flatten()
        except:
            # Older versions
            indices = indices.flatten() if isinstance(indices, np.ndarray) else indices
    
    # Process each detected object
    for i in indices:
        box = boxes[i]
        x, y, w, h = box
        
        # Ensure box is within image boundaries
        x = max(0, x)
        y = max(0, y)
        right = min(width - 1, x + w)
        bottom = min(height - 1, y + h)
        w = right - x
        h = bottom - y
        
        label = classes[class_ids[i]] if class_ids[i] < len(classes) else f"Unknown_{class_ids[i]}"
        
        detected_objects.append({
            "class_id": int(class_ids[i]),
            "label": label,
            "confidence": float(confidences[i]),
            "box": {
                "x": int(x),
                "y": int(y),
                "width": int(w),
                "height": int(h)
            },
            "center": {
                "x": int(x + w/2),
                "y": int(y + h/2)
            }
        })
        
        # Draw on the image for visualization
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = f"{label}: {confidences[i]:.2f}"
        cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Save annotated image if output path is provided
    if output_path:
        cv2.imwrite(output_path, image)
    
    return {
        "image_path": image_path,
        "output_path": output_path,
        "image_width": width,
        "image_height": height,
        "detection_count": len(detected_objects),
        "detected_objects": detected_objects
    } 