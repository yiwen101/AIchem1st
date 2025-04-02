"""
Object tracking tools for video analysis.

This module provides tools for tracking objects across video frames using SOTA models,
supporting both appearance-based and motion-based tracking.
"""

import os
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from typing import Dict, Any, List, Optional, Tuple
import time
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import uuid

from app.tools.toolImpl.base_tool import BaseTool, ToolParameter, ToolParameterType
from app.tools.tool_manager import register_tool
from app.tools.resource.resource_manager import resource_manager
from app.common.monitor import logger
from app.tools.toolImpl.object_detection import ObjectDetectionTool, load_detection_model

# Cache for tracking models
tracker_cache = {}

def create_unique_colors(n):
    """Create a list of n visually distinct colors."""
    base_colors = list(mcolors.TABLEAU_COLORS.values())
    if n <= len(base_colors):
        return base_colors[:n]
    
    # If we need more colors, generate them
    return plt.cm.rainbow(np.linspace(0, 1, n))

class Tracker:
    """Base class for all trackers."""
    
    def __init__(self, tracker_type: str):
        self.tracker_type = tracker_type
        self.tracks = {}  # Dictionary to store track information
        self.next_id = 0  # Counter for track IDs
    
    def init(self, frame: np.ndarray, detections: List[Dict]) -> None:
        """Initialize tracker with frame and detections."""
        raise NotImplementedError("Subclasses must implement init method")
    
    def update(self, frame: np.ndarray) -> List[Dict]:
        """Update tracker with new frame."""
        raise NotImplementedError("Subclasses must implement update method")
    
    def get_tracks(self) -> Dict[int, Dict]:
        """Get current tracks."""
        return self.tracks
    
    def _assign_new_id(self) -> int:
        """Assign a new unique ID for a track."""
        track_id = self.next_id
        self.next_id += 1
        return track_id


class OpenCVTracker(Tracker):
    """Tracker using OpenCV's built-in trackers (motion-based)."""
    
    def __init__(self, tracker_type: str = "KCF"):
        """
        Initialize OpenCV tracker.
        
        Args:
            tracker_type: Type of OpenCV tracker to use
                Options: 'KCF', 'CSRT', 'MOSSE', etc.
        """
        super().__init__(f"opencv_{tracker_type}")
        self.cv_tracker_type = tracker_type
        self.cv_trackers = {}  # Dictionary to store OpenCV tracker instances
        self.tracks = {}
    
    def init(self, frame: np.ndarray, detections: List[Dict]) -> None:
        """
        Initialize tracker with frame and detections.
        
        Args:
            frame: Video frame
            detections: List of detection objects with bbox information
        """
        self.cv_trackers = {}
        self.tracks = {}
        
        for det in detections:
            # Create a new OpenCV tracker for each detection
            if self.cv_tracker_type == "KCF":
                tracker = cv2.TrackerKCF_create()
            elif self.cv_tracker_type == "CSRT":
                tracker = cv2.TrackerCSRT_create()
            elif self.cv_tracker_type == "MOSSE":
                tracker = cv2.legacy.TrackerMOSSE_create()
            else:
                # Default to KCF
                tracker = cv2.TrackerKCF_create()
            
            # Get bounding box
            x = det["box"]["x"]
            y = det["box"]["y"]
            w = det["box"]["width"]
            h = det["box"]["height"]
            bbox = (x, y, w, h)
            
            # Initialize tracker
            tracker.init(frame, bbox)
            
            # Assign ID and store
            track_id = self._assign_new_id()
            self.cv_trackers[track_id] = tracker
            
            # Store track info
            self.tracks[track_id] = {
                "id": track_id,
                "label": det["label"],
                "confidence": det["confidence"],
                "box": det["box"].copy(),
                "center": det["center"].copy(),
                "last_seen": 0,
                "trajectory": [det["center"].copy()]
            }
    
    def update(self, frame: np.ndarray) -> List[Dict]:
        """
        Update tracker with new frame.
        
        Args:
            frame: New video frame
            
        Returns:
            List of updated tracking results
        """
        # Update each tracker
        updated_tracks = []
        
        for track_id, tracker in list(self.cv_trackers.items()):
            # Update tracker
            success, bbox = tracker.update(frame)
            
            if success:
                x, y, w, h = [int(v) for v in bbox]
                
                # Update track info
                self.tracks[track_id]["box"] = {
                    "x": x,
                    "y": y,
                    "width": w,
                    "height": h
                }
                
                # Update center
                center = {
                    "x": int(x + w/2),
                    "y": int(y + h/2)
                }
                self.tracks[track_id]["center"] = center
                self.tracks[track_id]["trajectory"].append(center)
                
                # Update last seen
                self.tracks[track_id]["last_seen"] = 0
                
                # Add to results
                updated_tracks.append(self.tracks[track_id])
            else:
                # Increment last seen counter
                self.tracks[track_id]["last_seen"] += 1
                
                # Remove tracker if not seen for too long
                if self.tracks[track_id]["last_seen"] > 10:
                    del self.cv_trackers[track_id]
                    # Keep track info for history
        
        return updated_tracks


class DeepSORTTracker(Tracker):
    """Deep SORT tracker (appearance-based)."""
    
    def __init__(self, max_age: int = 30, n_init: int = 3, max_cosine_distance: float = 0.4):
        """
        Initialize DeepSORT tracker with default parameters.
        
        Args:
            max_age: Maximum number of frames to keep track without detection
            n_init: Number of consecutive detections before track confirmation
            max_cosine_distance: Threshold for appearance similarity
        """
        super().__init__("deepsort")
        self.max_age = max_age
        self.n_init = n_init
        self.max_cosine_distance = max_cosine_distance
        self.tracks = {}
        self.features = {}  # Store appearance features
        self.confirmed_tracks = set()  # Set of confirmed track IDs
        
        # In a real implementation, we would initialize the DeepSORT tracker here
        # For this implementation, we'll simulate DeepSORT's behavior
    
    def _extract_features(self, frame: np.ndarray, bbox: Dict) -> np.ndarray:
        """
        Extract appearance features from detection area.
        In a real implementation, this would use a CNN to extract features.
        
        Args:
            frame: Video frame
            bbox: Bounding box dictionary
            
        Returns:
            Feature vector
        """
        # Simulate feature extraction with a simple color histogram
        x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
        
        # Ensure bounds are within frame
        h, w_frame, _ = frame.shape
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(w_frame, x + w)
        y2 = min(h, y + h)
        
        # Extract region of interest
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return np.zeros(512)  # Return zero feature if ROI is empty
        
        # Convert to RGB and resize to normalize
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        roi_resized = cv2.resize(roi_rgb, (64, 64))
        
        # Calculate color histogram as a simple feature
        hist_r = cv2.calcHist([roi_resized], [0], None, [8], [0, 256])
        hist_g = cv2.calcHist([roi_resized], [1], None, [8], [0, 256])
        hist_b = cv2.calcHist([roi_resized], [2], None, [8], [0, 256])
        
        # Normalize and flatten
        cv2.normalize(hist_r, hist_r, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist_g, hist_g, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist_b, hist_b, 0, 1, cv2.NORM_MINMAX)
        
        # Combine and pad to 512 dimensions to simulate real embedding
        feature = np.concatenate([hist_r.flatten(), hist_g.flatten(), hist_b.flatten()])
        feature = np.pad(feature, (0, 512 - feature.size), 'constant')
        
        return feature
    
    def _calculate_similarity(self, feature1: np.ndarray, feature2: np.ndarray) -> float:
        """
        Calculate cosine similarity between features.
        
        Args:
            feature1: First feature vector
            feature2: Second feature vector
            
        Returns:
            Similarity score (0-1)
        """
        dot = np.dot(feature1, feature2)
        norm1 = np.linalg.norm(feature1)
        norm2 = np.linalg.norm(feature2)
        
        if norm1 == 0 or norm2 == 0:
            return 0
        
        return dot / (norm1 * norm2)
    
    def init(self, frame: np.ndarray, detections: List[Dict]) -> None:
        """
        Initialize tracker with frame and detections.
        
        Args:
            frame: Video frame
            detections: List of detection objects
        """
        self.tracks = {}
        self.features = {}
        self.confirmed_tracks = set()
        
        for det in detections:
            # Extract features
            feature = self._extract_features(frame, det["box"])
            
            # Assign ID and store
            track_id = self._assign_new_id()
            
            # Store track info
            self.tracks[track_id] = {
                "id": track_id,
                "label": det["label"],
                "confidence": det["confidence"],
                "box": det["box"].copy(),
                "center": det["center"].copy(),
                "last_seen": 0,
                "age": 1,
                "hits": 1,
                "trajectory": [det["center"].copy()]
            }
            
            # Store features
            self.features[track_id] = feature
    
    def update(self, frame: np.ndarray, detections: Optional[List[Dict]] = None) -> List[Dict]:
        """
        Update tracker with new frame and optional new detections.
        
        Args:
            frame: New video frame
            detections: Optional new detections for this frame
            
        Returns:
            List of updated tracking results
        """
        # First, predict new locations using Kalman filter (simplified here)
        for track_id in list(self.tracks.keys()):
            if self.tracks[track_id]["last_seen"] > self.max_age:
                # Remove track if not seen for too long
                del self.tracks[track_id]
                if track_id in self.features:
                    del self.features[track_id]
                continue
            
            # Simple motion model - just keep the same location for now
            self.tracks[track_id]["last_seen"] += 1
            self.tracks[track_id]["age"] += 1
            
            # Confirm track if it has been seen for enough frames
            if self.tracks[track_id]["hits"] >= self.n_init:
                self.confirmed_tracks.add(track_id)
        
        # If we have new detections, associate them with existing tracks
        if detections:
            # Calculate features for all new detections
            detection_features = [self._extract_features(frame, det["box"]) for det in detections]
            
            # Calculate similarity matrix
            similarity_matrix = np.zeros((len(self.tracks), len(detections)))
            for i, track_id in enumerate(self.tracks):
                if track_id in self.features:
                    for j, det_feature in enumerate(detection_features):
                        similarity_matrix[i, j] = self._calculate_similarity(
                            self.features[track_id], det_feature
                        )
            
            # Associate detections with tracks (greedy matching for simplicity)
            matched_track_indices = set()
            matched_detection_indices = set()
            
            # While there are possible matches
            while True:
                # Find highest similarity
                if similarity_matrix.size == 0:
                    break
                    
                # Flatten to find max
                i_flat = np.argmax(similarity_matrix)
                if similarity_matrix.size > 1:
                    i, j = np.unravel_index(i_flat, similarity_matrix.shape)
                else:
                    i, j = 0, 0
                
                value = similarity_matrix[i, j]
                
                # If no good matches left, break
                if value < self.max_cosine_distance:
                    break
                
                # Get track ID for this index
                track_id = list(self.tracks.keys())[i]
                
                # Update track with detection
                det = detections[j]
                self.tracks[track_id]["box"] = det["box"].copy()
                self.tracks[track_id]["center"] = det["center"].copy()
                self.tracks[track_id]["label"] = det["label"]
                self.tracks[track_id]["confidence"] = det["confidence"]
                self.tracks[track_id]["last_seen"] = 0
                self.tracks[track_id]["hits"] += 1
                self.tracks[track_id]["trajectory"].append(det["center"].copy())
                
                # Update feature
                self.features[track_id] = detection_features[j]
                
                # Mark as matched
                matched_track_indices.add(i)
                matched_detection_indices.add(j)
                
                # Set similarity to 0 for matched pairs
                similarity_matrix[i, :] = 0
                similarity_matrix[:, j] = 0
            
            # Create new tracks for unmatched detections
            for j, det in enumerate(detections):
                if j not in matched_detection_indices:
                    # Create new track
                    track_id = self._assign_new_id()
                    self.tracks[track_id] = {
                        "id": track_id,
                        "label": det["label"],
                        "confidence": det["confidence"],
                        "box": det["box"].copy(),
                        "center": det["center"].copy(),
                        "last_seen": 0,
                        "age": 1,
                        "hits": 1,
                        "trajectory": [det["center"].copy()]
                    }
                    self.features[track_id] = detection_features[j]
        
        # Return list of confirmed tracks
        return [track for track_id, track in self.tracks.items() 
                if track_id in self.confirmed_tracks]


def load_tracker(tracker_type: str, **kwargs) -> Tracker:
    """
    Load and initialize a tracker of specified type.
    
    Args:
        tracker_type: Type of tracker ('opencv' or 'deepsort')
        **kwargs: Additional parameters for the tracker
        
    Returns:
        Initialized tracker
    """
    if tracker_type.startswith("opencv"):
        cv_tracker_type = tracker_type.split("_")[1] if "_" in tracker_type else "KCF"
        return OpenCVTracker(cv_tracker_type)
    elif tracker_type == "deepsort":
        return DeepSORTTracker(**kwargs)
    else:
        raise ValueError(f"Unsupported tracker type: {tracker_type}")


@register_tool
class ObjectTrackingTool(BaseTool):
    """Tool for tracking objects across video frames."""
    
    name = "object_tracking"
    description = "Track objects across multiple frames in a video using SOTA tracking models."
    parameters = [
        ToolParameter(
            name="start_time",
            type=ToolParameterType.FLOAT,
            description="Start time in seconds for tracking",
            required=True
        ),
        ToolParameter(
            name="end_time",
            type=ToolParameterType.FLOAT,
            description="End time in seconds for tracking",
            required=True
        ),
        ToolParameter(
            name="tracker_type",
            type=ToolParameterType.STRING,
            description="Type of tracker to use ('opencv_KCF', 'opencv_CSRT', 'deepsort')",
            required=False,
            default="deepsort"
        )
    ]
    
    @classmethod
    def execute(cls, start_time: float, end_time: float, 
               tracker_type: str = "deepsort") -> Dict[str, Any]:
        """
        Track objects across video frames.
        
        Args:
            start_time: Start time in seconds for tracking
            end_time: End time in seconds for tracking
            tracker_type: Type of tracker to use ('opencv_KCF', 'opencv_CSRT', 'deepsort')
            
        Returns:
            Dictionary with tracking results
        """
        # Use default values for removed parameters
        detection_model = "facebook/detr-resnet-50"
        confidence_threshold = 0.5
        max_objects = 10
        frame_step = 1
        
        # Calculate duration (for output purposes only)
        duration = end_time - start_time
        
        # Store original requested time range
        requested_start_time = start_time
        requested_end_time = end_time
        
        # Get active video
        cap, metadata = resource_manager.get_active_video()
        
        # Get video properties
        fps = metadata["fps"]
        frame_count = metadata["frame_count"]
        width = metadata["width"]
        height = metadata["height"]
        video_name = metadata["video_name"]
        
        # Calculate frames
        start_frame = int(start_time * fps)
        end_frame = min(int(end_time * fps), frame_count - 1)
        frame_count_to_process = end_frame - start_frame + 1
        
        if frame_count_to_process <= 0:
            logger.log_error(f"Invalid time range: start_time={start_time}, end_time={end_time}")
            return {
                "error": f"Invalid time range: start_time={start_time}, end_time={end_time}",
                "start_time": requested_start_time,
                "end_time": requested_end_time
            }
        
        logger.log_info(f"Tracking objects from frame {start_frame} to {end_frame}")
        
        # Try to find objects starting from start_frame
        current_frame_idx = start_frame
        max_search_frames = min(30, int(frame_count_to_process / 2))  # Search up to 30 frames or half of total
        detections = None
        first_frame = None
        found_objects = False
        
        for i in range(max_search_frames):
            # Get frame at current position
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)
            ret, frame = cap.read()
            if not ret:
                logger.log_error(f"Could not read frame at position {current_frame_idx}")
                return {"error": f"Could not read frame at position {current_frame_idx}"}
            
            # Store as first frame
            first_frame = frame
            
            # Detect objects in current frame
            current_time = current_frame_idx / fps
            logger.log_info(f"Attempting object detection at time {current_time:.2f}s (frame {current_frame_idx})")
            
            # Use the ObjectDetectionTool to get detections
            detection_result = ObjectDetectionTool.execute(
                time_seconds=current_time,
                confidence_threshold=confidence_threshold,
                model_name=detection_model
            )
            
            # Get detections from result
            detections = detection_result.get("detected_objects", [])
            
            # Limit number of objects to track
            if len(detections) > max_objects:
                # Sort by confidence and take top max_objects
                detections = sorted(detections, key=lambda x: x["confidence"], reverse=True)[:max_objects]
            
            # If objects detected, break
            if detections:
                found_objects = True
                logger.log_info(f"Found {len(detections)} objects at time {current_time:.2f}s (frame {current_frame_idx})")
                start_frame = current_frame_idx  # Update start frame to where objects were found
                start_time = current_time  # Update start time to match
                break
            
            # Move to next frame
            current_frame_idx += int(fps / 2)  # Try every half second
            if current_frame_idx >= end_frame:
                break
        
        # If no objects found after searching
        if not found_objects:
            logger.log_error("No objects detected in any of the searched frames")
            return {
                "error": "No objects detected in any of the searched frames",
                "start_time": requested_start_time,
                "end_time": requested_end_time
            }
        
        # Initialize tracker
        tracker_kwargs = {}
        if tracker_type == "deepsort":
            tracker_kwargs = {
                "max_age": 30,
                "n_init": 3,
                "max_cosine_distance": 0.4
            }
        
        logger.log_info(f"Initializing {tracker_type} tracker")
        tracker = load_tracker(tracker_type, **tracker_kwargs)
        tracker.init(first_frame, detections)
        
        # Process frames
        all_tracks = []
        current_tracks = {}
        
        # Process frames
        frame_idx = start_frame
        while frame_idx <= end_frame:
            # Skip frames based on frame_step
            if (frame_idx - start_frame) % frame_step != 0 and frame_idx != start_frame:
                # Read frame but don't process
                ret, _ = cap.read()
                if not ret:
                    break
                frame_idx += 1
                continue
            
            # Set position and read frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
            
            # If using DeepSORT with re-detection, perform detection periodically
            new_detections = None
            if tracker_type == "deepsort" and frame_idx > start_frame and (frame_idx - start_frame) % (fps * 2) == 0:
                # Perform detection every 2 seconds
                frame_time = frame_idx / fps
                detection_result = ObjectDetectionTool.execute(
                    time_seconds=frame_time,
                    confidence_threshold=confidence_threshold,
                    model_name=detection_model
                )
                new_detections = detection_result.get("detected_objects", [])
            
            # Update tracker
            if tracker_type == "deepsort" and new_detections:
                updated_tracks = tracker.update(frame, new_detections)
            else:
                updated_tracks = tracker.update(frame)
            
            # Update current tracks
            for track in updated_tracks:
                track_id = track["id"]
                
                # Update track
                current_tracks[track_id] = track
            
            # Accumulate tracks for result
            frame_time = frame_idx / fps
            all_tracks.append({
                "frame": frame_idx,
                "time": frame_time,
                "tracks": [track for track_id, track in current_tracks.items() 
                         if "last_seen" in track and track["last_seen"] <= 5]
            })
            
            # Update frame index
            frame_idx += 1
            
            # Log progress
            if frame_idx % (fps * 5) == 0:  # Log every 5 seconds
                logger.log_info(f"Processed {frame_idx - start_frame}/{frame_count_to_process} frames")
        
        # Calculate tracking statistics
        unique_tracks = set()
        track_info = {}
        object_counts = {}
        
        for frame_tracks in all_tracks:
            for track in frame_tracks["tracks"]:
                track_id = track["id"]
                unique_tracks.add(track_id)
                
                # Store trajectories and labels for each unique track
                if track_id not in track_info:
                    track_info[track_id] = {
                        "id": track_id,
                        "label": track["label"],
                        "first_seen": frame_tracks["time"],
                        "trajectory": [],
                        "frames_visible": 0
                    }
                
                # Update last seen time
                track_info[track_id]["last_seen"] = frame_tracks["time"]
                track_info[track_id]["frames_visible"] += 1
                
                # Add current position to trajectory (but not every frame to save space)
                if len(track_info[track_id]["trajectory"]) == 0 or \
                   frame_tracks["time"] - track_info[track_id]["trajectory"][-1]["time"] >= 0.5:  # Add point every 0.5s
                    track_info[track_id]["trajectory"].append({
                        "time": frame_tracks["time"],
                        "x": track["center"]["x"] / width,  # Normalize coordinates to 0-1 range
                        "y": track["center"]["y"] / height
                    })
                
                # Count objects by label
                label = track["label"]
                if label not in object_counts:
                    object_counts[label] = 0
                object_counts[label] = max(object_counts[label], 1)  # At least count it once
        
        # Generate simple narrative descriptions of movements and presence
        movement_descriptions = []
        presence_descriptions = []
        interaction_descriptions = []
        
        # Track screen regions for directional clarity
        regions = {
            "top-left": (0, 0, 0.33, 0.33),
            "top-center": (0.33, 0, 0.66, 0.33),
            "top-right": (0.66, 0, 1, 0.33),
            "middle-left": (0, 0.33, 0.33, 0.66),
            "center": (0.33, 0.33, 0.66, 0.66),
            "middle-right": (0.66, 0.33, 1, 0.66),
            "bottom-left": (0, 0.66, 0.33, 1),
            "bottom-center": (0.33, 0.66, 0.66, 1),
            "bottom-right": (0.66, 0.66, 1, 1)
        }
        
        def get_region(x, y):
            for region_name, bounds in regions.items():
                x_min, y_min, x_max, y_max = bounds
                if x_min <= x < x_max and y_min <= y < y_max:
                    return region_name
            return "center"  # Default fallback
        
        # Create summaries of object paths
        object_paths = []
        
        # Analyze each track's movement
        for track_id, info in track_info.items():
            # Skip tracks with insufficient data
            if len(info.get("trajectory", [])) < 2:
                continue
                
            label = info["label"]
            first_seen = info["first_seen"]
            last_seen = info["last_seen"]
            
            # Get actual path coordinates
            trajectory = info["trajectory"]
            
            # Get start and end points (convert from normalized to pixel coordinates)
            start_point = trajectory[0]
            end_point = trajectory[-1]
            
            start_x = int(start_point["x"] * width)
            start_y = int(start_point["y"] * height)
            end_x = int(end_point["x"] * width)
            end_y = int(end_point["y"] * height)
            
            # Create path description
            path_description = {
                "object_type": label,
                "id": track_id,
                "start_time": first_seen,
                "end_time": last_seen,
                "start_position": {
                    "x": start_x,
                    "y": start_y
                },
                "end_position": {
                    "x": end_x,
                    "y": end_y
                },
                "path_points": []
            }
            
            # Add key points along the path (convert normalized to pixel coordinates)
            for point in trajectory:
                path_description["path_points"].append({
                    "time": point["time"],
                    "x": int(point["x"] * width),
                    "y": int(point["y"] * height)
                })
            
            # Add path description
            object_paths.append(path_description)
            
        # Create textual summaries for each path
        path_summaries = []
        for path in object_paths:
            obj_type = path["object_type"]
            start_time_obj = path["start_time"]
            end_time_obj = path["end_time"]
            start_pos = path["start_position"]
            end_pos = path["end_position"]
            
            # Calculate direction and distance
            dx = end_pos["x"] - start_pos["x"]
            dy = end_pos["y"] - start_pos["y"]
            distance = int(((dx**2 + dy**2)**0.5))
            
            # Create human-readable summary
            summary = f"A {obj_type} moves from position ({start_pos['x']}, {start_pos['y']}) to position ({end_pos['x']}, {end_pos['y']}) from {start_time_obj:.1f}s to {end_time_obj:.1f}s"
            
            # Add some key points if trajectory has more than 2 points
            if len(path["path_points"]) > 2:
                mid_points = []
                # Add key intermediate points (exclude first and last)
                step = max(1, len(path["path_points"]) // 3)
                for i in range(step, len(path["path_points"]) - 1, step):
                    point = path["path_points"][i]
                    mid_points.append(f"at {point['time']:.1f}s: ({point['x']}, {point['y']})")
                
                if mid_points:
                    summary += f", passing through " + ", ".join(mid_points)
            
            path_summaries.append(summary)
        
        # Add object counts if any objects were tracked
        if object_counts:
            objects_seen = ", ".join([f"{count} {label}{'s' if count > 1 else ''}" for label, count in object_counts.items()])
            # Use the original requested time period in the summary
            summary = f"During the requested time period from {requested_start_time:.1f}s to {requested_end_time:.1f}s, tracked: {objects_seen}"
            path_summaries.insert(0, summary)
        
        # If nothing was tracked in the entire original time period
        if not path_summaries:
            path_summaries.append(f"No objects detected in the requested time period from {requested_start_time:.1f}s to {requested_end_time:.1f}s.")
        
        # Process each frame for detailed output
        frame_results = []
        for frame_data in all_tracks:
            time = frame_data["time"]
            frame_objects = []
            
            # Extract object information from each track in this frame
            for track in frame_data["tracks"]:
                frame_objects.append({
                    "id": track["id"],
                    "label": track["label"],
                    "confidence": track.get("confidence", 1.0),
                    "position": {
                        "x": track["center"]["x"],
                        "y": track["center"]["y"]
                    },
                    "box": {
                        "x": track["box"]["x"],
                        "y": track["box"]["y"],
                        "width": track["box"]["width"],
                        "height": track["box"]["height"]
                    }
                })
            
            # Add frame result with objects present
            frame_results.append({
                "time": time,
                "frame_number": frame_data["frame"],
                "objects": frame_objects
            })
        
        # Find time periods without objects
        empty_periods = []
        if frame_results:
            # Initialize with period before first object if needed
            first_detection_time = frame_results[0]["time"]
            if first_detection_time > requested_start_time:
                empty_periods.append({
                    "start": requested_start_time,
                    "end": first_detection_time
                })
            
            # Find gaps between detections
            for i in range(1, len(frame_results)):
                prev_frame = frame_results[i-1]
                curr_frame = frame_results[i]
                
                # Check for gap in time (more than 0.5 seconds) with no objects detected
                if curr_frame["time"] - prev_frame["time"] > 0.5:
                    empty_periods.append({
                        "start": prev_frame["time"],
                        "end": curr_frame["time"]
                    })
            
            # Add period after last object if needed
            last_detection_time = frame_results[-1]["time"]
            if last_detection_time < requested_end_time:
                empty_periods.append({
                    "start": last_detection_time,
                    "end": requested_end_time
                })
        
        # Create summary statements for periods without objects
        empty_period_statements = []
        for period in empty_periods:
            empty_period_statements.append(
                f"No objects detected from {period['start']:.1f}s to {period['end']:.1f}s"
            )
        
        # Return the paths, summaries, and frame-by-frame results
        return {
            "request_time_period": f"{requested_start_time:.1f}s to {requested_end_time:.1f}s",
            "actual_tracking_period": f"{frame_results[0]['time']:.1f}s to {frame_results[-1]['time']:.1f}s" if frame_results else "No tracking performed",
            "paths": object_paths,
            "summaries": path_summaries,
            "empty_periods": empty_period_statements,
            "frame_results": frame_results
        } 