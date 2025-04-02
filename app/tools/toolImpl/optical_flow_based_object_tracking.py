"""
Optical flow based object tracking using RAFT model.

This module provides a tracker that uses optical flow to track objects across frames,
utilizing the RAFT (Recurrent All-Pairs Field Transforms) model for accurate flow estimation.
"""

import os
import cv2
import numpy as np
import torch
from typing import Dict, Any, List, Optional, Tuple
import time
import matplotlib.pyplot as plt

from app.tools.toolImpl.base_tool import BaseTool, ToolParameter, ToolParameterType
from app.tools.tool_manager import register_tool
from app.tools.resource.resource_manager import resource_manager
from app.common.monitor import logger

class OpticalFlowTracker:
    """
    Optical flow based tracker using RAFT model.
    
    This tracker uses optical flow to track features across frames of a video.
    """
    
    def __init__(self, use_cuda: bool = False):
        """
        Initialize the OpticalFlowTracker.
        
        Args:
            use_cuda: Whether to use CUDA for acceleration
        """
        self.tracks = {}  # Dictionary to store track information
        self.next_id = 0  # Counter for track IDs
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')
        
        # Initialize the optical flow model (RAFT)
        self._init_flow_model()
        
        # Parameters for optical flow calculation
        self.flow_params = {
            'pyr_scale': 0.5,
            'levels': 3,
            'winsize': 15,
            'iterations': 3,
            'poly_n': 5,
            'poly_sigma': 1.2,
            'flags': 0
        }
        
        # For tracking
        self.prev_gray = None
        self.prev_features = None
        self.tracking_features = None
        
        logger.log_info(f"Initialized OpticalFlowTracker (CUDA: {self.use_cuda})")
    
    def _init_flow_model(self):
        """Initialize the optical flow model."""
        try:
            # First try to use RAFT model if available
            import torch
            
            # Set default to not using RAFT
            self.use_raft = False
            
            # Check if PyTorch is available with CUDA if requested
            if not torch.cuda.is_available() and self.use_cuda:
                logger.log_warning("CUDA requested but not available, falling back to CPU")
                self.use_cuda = False
            
            logger.log_info(f"PyTorch detected, device: {self.device}")
            
            # Try multiple methods to load RAFT
            try:
                # Method 1: Try loading from torch hub
                if hasattr(torch.hub, 'load'):
                    try:
                        logger.log_info("Attempting to load RAFT model from torch hub")
                        self.flow_model = torch.hub.load('pytorch/vision:v0.10.0', 'raft_small', pretrained=True)
                        self.flow_model.to(self.device)
                        self.flow_model.eval()
                        self.use_raft = True
                        logger.log_info(f"Successfully loaded RAFT model via torch hub (device: {self.device})")
                        return
                    except Exception as e:
                        logger.log_warning(f"Failed to load RAFT from torch hub: {e}")
                
                # Method 2: Try loading from torchvision if available
                try:
                    import torchvision
                    if hasattr(torchvision.models, 'optical_flow') and hasattr(torchvision.models.optical_flow, 'raft_small'):
                        logger.log_info("Attempting to load RAFT model from torchvision")
                        self.flow_model = torchvision.models.optical_flow.raft_small(pretrained=True)
                        self.flow_model.to(self.device)
                        self.flow_model.eval()
                        self.use_raft = True
                        logger.log_info(f"Successfully loaded RAFT model via torchvision (device: {self.device})")
                        return
                except ImportError:
                    logger.log_warning("Torchvision not available or doesn't have RAFT model")
                except Exception as e:
                    logger.log_warning(f"Failed to load RAFT from torchvision: {e}")
            
            except Exception as e:
                logger.log_warning(f"Error during RAFT model loading attempts: {e}")
            
            logger.log_warning("All attempts to load RAFT model failed")
            
        except ImportError:
            logger.log_warning("PyTorch not available, falling back to OpenCV optical flow")
        
        # If RAFT not available or failed to load, fall back to OpenCV's optical flow
        logger.log_info("Using OpenCV's optical flow as fallback")
        self.flow_model = None
        self.use_raft = False
    
    def _calculate_flow_raft(self, prev_frame: np.ndarray, current_frame: np.ndarray) -> np.ndarray:
        """
        Calculate optical flow using RAFT model.
        
        Args:
            prev_frame: Previous frame
            current_frame: Current frame
            
        Returns:
            Flow field (u, v) of shape [H, W, 2]
        """
        try:
            # Convert frames to RGB
            prev_rgb = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2RGB)
            curr_rgb = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
            
            # Get original dimensions
            orig_h, orig_w = prev_rgb.shape[:2]
            
            # Calculate padding to make dimensions divisible by 8
            pad_h = (8 - orig_h % 8) % 8
            pad_w = (8 - orig_w % 8) % 8
            
            # Apply padding if needed
            if pad_h > 0 or pad_w > 0:
                # Create padded images with zeros
                prev_rgb_padded = np.pad(prev_rgb, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
                curr_rgb_padded = np.pad(curr_rgb, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
            else:
                prev_rgb_padded = prev_rgb
                curr_rgb_padded = curr_rgb
            
            # Convert to PyTorch tensors
            prev_tensor = torch.from_numpy(prev_rgb_padded).float().permute(2, 0, 1).unsqueeze(0) / 255.0
            curr_tensor = torch.from_numpy(curr_rgb_padded).float().permute(2, 0, 1).unsqueeze(0) / 255.0
            
            # Move to device
            prev_tensor = prev_tensor.to(self.device)
            curr_tensor = curr_tensor.to(self.device)
            
            # Calculate flow
            with torch.no_grad():
                # Get flow predictions from RAFT
                flow_output = self.flow_model(prev_tensor, curr_tensor)
                
                # Handle different output formats from RAFT
                # Some versions return a tuple of predictions, others return a dictionary
                if isinstance(flow_output, tuple):
                    # Get the final prediction (usually the last element)
                    flow_pred = flow_output[-1]
                elif isinstance(flow_output, dict):
                    # Check if 'flows' key exists
                    if 'flows' in flow_output:
                        flow_pred = flow_output['flows'][-1]  # Get final prediction
                    else:
                        # Try to find any tensor with shape [B,2,H,W]
                        for k, v in flow_output.items():
                            if isinstance(v, torch.Tensor) and len(v.shape) == 4 and v.shape[1] == 2:
                                flow_pred = v
                                break
                        else:
                            raise ValueError(f"Could not find flow prediction in output: {flow_output.keys()}")
                elif isinstance(flow_output, list):
                    # Some RAFT implementations return a list of flow predictions at different scales
                    # Take the last one which is usually the full resolution prediction
                    flow_pred = flow_output[-1]
                    logger.log_info(f"RAFT returned a list of predictions, using the last one")
                else:
                    # Assume it's already the flow tensor
                    flow_pred = flow_output
                
                # Handle if flow_pred is a list
                if isinstance(flow_pred, list):
                    logger.log_info(f"Flow prediction is a list with {len(flow_pred)} elements")
                    # Take the last element if it's a list
                    if len(flow_pred) > 0:
                        flow_pred = flow_pred[-1]
                    else:
                        raise ValueError("Empty flow prediction list")
                
                # Convert from [B,2,H,W] to [H,W,2] format
                if len(flow_pred.shape) == 4 and flow_pred.shape[1] == 2:
                    flow_padded = flow_pred[0].permute(1, 2, 0).cpu().numpy()
                else:
                    # Try to handle other tensor shapes
                    if len(flow_pred.shape) == 3 and flow_pred.shape[0] == 2:
                        # If shape is [2,H,W]
                        flow_padded = flow_pred.permute(1, 2, 0).cpu().numpy()
                    else:
                        logger.log_error(f"Unexpected flow tensor shape: {flow_pred.shape}")
                        raise ValueError(f"Unexpected flow shape: {flow_pred.shape}")
                
                # Crop flow back to original dimensions
                flow = flow_padded[:orig_h, :orig_w, :]
                
                return flow
                
        except Exception as e:
            logger.log_error(f"Error in RAFT flow calculation: {str(e)}. Falling back to OpenCV flow.")
            # Fall back to OpenCV flow calculation
            # Ensure we're using RGB frames for OpenCV
            if prev_frame.ndim == 2:  # If already grayscale
                prev_gray = prev_frame
            else:
                prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                
            if current_frame.ndim == 2:  # If already grayscale
                curr_gray = current_frame
            else:
                curr_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
                
            return self._calculate_flow_cv(prev_gray, curr_gray)
    
    def _calculate_flow_cv(self, prev_gray: np.ndarray, current_gray: np.ndarray) -> np.ndarray:
        """
        Calculate optical flow using OpenCV's Farneback method.
        
        Args:
            prev_gray: Previous frame in grayscale
            current_gray: Current frame in grayscale
            
        Returns:
            Flow field (u, v) of shape [H, W, 2]
        """
        # Calculate dense optical flow
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, current_gray, None, 
            self.flow_params['pyr_scale'],
            self.flow_params['levels'],
            self.flow_params['winsize'],
            self.flow_params['iterations'],
            self.flow_params['poly_n'],
            self.flow_params['poly_sigma'],
            self.flow_params['flags']
        )
        
        return flow
    
    def _assign_new_id(self) -> int:
        """Assign a new unique ID for a track."""
        track_id = self.next_id
        self.next_id += 1
        return track_id
    
    def detect_features(self, frame: np.ndarray, max_features: int = 100) -> List[Dict]:
        """
        Detect features to track in the frame.
        
        Args:
            frame: Video frame
            max_features: Maximum number of features to detect
            
        Returns:
            List of feature dictionaries
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect features
        features = cv2.goodFeaturesToTrack(
            gray, maxCorners=max_features, qualityLevel=0.01, 
            minDistance=10, blockSize=7
        )
        
        if features is None:
            return []
        
        # Initialize tracks for each feature
        tracked_features = []
        for i, feature in enumerate(features):
            x, y = feature.ravel()
            
            # Create a region around the feature point
            feature_size = 20  # Size of the feature box
            x_min = max(0, int(x - feature_size/2))
            y_min = max(0, int(y - feature_size/2))
            width = min(feature_size, frame.shape[1] - x_min)
            height = min(feature_size, frame.shape[0] - y_min)
            
            track_id = self._assign_new_id()
            
            # Assign a generic label based on position
            # This replaces the object detection labels
            if y < frame.shape[0] / 3:
                label = "top_region_point"
            elif y < 2 * frame.shape[0] / 3:
                label = "middle_region_point"
            else:
                label = "bottom_region_point"
                
            if x < frame.shape[1] / 3:
                label = f"left_{label}"
            elif x < 2 * frame.shape[1] / 3:
                label = f"center_{label}"
            else:
                label = f"right_{label}"
            
            tracked_features.append({
                "id": track_id,
                "label": label,
                "confidence": 1.0,  # No actual confidence score for features
                "box": {
                    "x": x_min,
                    "y": y_min,
                    "width": width,
                    "height": height
                },
                "center": {"x": int(x), "y": int(y)},
                "last_seen": 0,
                "trajectory": [{"x": int(x), "y": int(y)}],
                "points": np.array([[[x, y]]], dtype=np.float32)
            })
        
        return tracked_features
    
    def init(self, frame: np.ndarray) -> None:
        """
        Initialize tracker with frame.
        
        Args:
            frame: Video frame
        """
        # Reset tracks
        self.tracks = {}
        
        # Store original frame for optical flow
        # RAFT uses color, OpenCV uses grayscale
        self.prev_gray = frame.copy()
        
        # Detect features to track
        features = self.detect_features(frame)
        
        # Initialize tracks
        for feature in features:
            track_id = feature["id"]
            self.tracks[track_id] = feature
    
    def update(self, frame: np.ndarray) -> List[Dict]:
        """
        Update tracker with new frame.
        
        Args:
            frame: New video frame
            
        Returns:
            List of updated tracking results
        """
        # Convert to grayscale for OpenCV flow (RAFT will use color directly)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate optical flow
        try:
            if self.use_raft and self.flow_model is not None:
                # For RAFT, we use the original BGR frames
                flow = self._calculate_flow_raft(self.prev_gray, frame)
            else:
                # For OpenCV, we use the grayscale frames
                prev_gray = self.prev_gray if self.prev_gray.ndim == 2 else cv2.cvtColor(self.prev_gray, cv2.COLOR_BGR2GRAY)
                flow = self._calculate_flow_cv(prev_gray, gray)
        except Exception as e:
            logger.log_error(f"Error calculating flow: {str(e)}. Falling back to OpenCV flow.")
            # Ensure we have grayscale images for OpenCV
            if self.prev_gray.ndim != 2:
                prev_gray = cv2.cvtColor(self.prev_gray, cv2.COLOR_BGR2GRAY)
            flow = self._calculate_flow_cv(prev_gray, gray)
        
        # Update tracks
        updated_tracks = []
        
        for track_id, track in list(self.tracks.items()):
            # Skip if not seen for too long
            if track["last_seen"] > 10:
                continue
            
            # Get points to track
            points = track["points"]
            
            # Track points using the flow field
            new_points = []
            valid_points = 0
            
            for p in points:
                px, py = int(p[0][0]), int(p[0][1])
                
                # Check if point is within frame bounds
                if 0 <= px < flow.shape[1] and 0 <= py < flow.shape[0]:
                    # Get flow at point
                    dx, dy = flow[py, px]
                    
                    # Calculate new position
                    new_x = px + dx
                    new_y = py + dy
                    
                    # Check if new position is within frame
                    if 0 <= new_x < frame.shape[1] and 0 <= new_y < frame.shape[0]:
                        new_points.append([[new_x, new_y]])
                        valid_points += 1
            
            # Convert new points to numpy array
            if new_points:
                new_points = np.array(new_points, dtype=np.float32)
                
                # Calculate new bounding box based on tracked points
                if len(new_points) >= 1:
                    min_x = min(p[0][0] for p in new_points)
                    min_y = min(p[0][1] for p in new_points)
                    max_x = max(p[0][0] for p in new_points)
                    max_y = max(p[0][1] for p in new_points)
                    
                    # Calculate width and height
                    w = max(5, max_x - min_x)
                    h = max(5, max_y - min_y)
                    
                    # Adjust aspect ratio to be similar to original box if possible
                    orig_w = track["box"]["width"]
                    orig_h = track["box"]["height"]
                    orig_ratio = orig_w / orig_h if orig_h > 0 else 1.0
                    new_ratio = w / h if h > 0 else 1.0
                    
                    if new_ratio > orig_ratio * 1.5:
                        h = w / orig_ratio
                    elif new_ratio < orig_ratio / 1.5:
                        w = h * orig_ratio
                    
                    # Update track
                    x = max(0, min_x)
                    y = max(0, min_y)
                    
                    # Calculate center
                    center_x = x + w / 2
                    center_y = y + h / 2
                    
                    # Update box
                    track["box"] = {
                        "x": int(x),
                        "y": int(y),
                        "width": int(w),
                        "height": int(h)
                    }
                    
                    # Update center
                    track["center"] = {
                        "x": int(center_x),
                        "y": int(center_y)
                    }
                    
                    # Update trajectory
                    track["trajectory"].append(track["center"].copy())
                    
                    # Update tracking points
                    track["points"] = new_points
                    
                    # Reset last seen
                    track["last_seen"] = 0
                    
                    # Add to results
                    updated_tracks.append(track)
                else:
                    # Not enough points tracked
                    track["last_seen"] += 1
            else:
                # No points tracked
                track["last_seen"] += 1
        
        # Detect new features if we're tracking fewer than desired
        if len(updated_tracks) < 20:
            # Detect new features
            new_features = self.detect_features(frame, max_features=20 - len(updated_tracks))
            
            # Add to tracks
            for feature in new_features:
                track_id = feature["id"]
                self.tracks[track_id] = feature
                updated_tracks.append(feature)
        
        # Update previous frame
        self.prev_gray = gray
        
        return updated_tracks


@register_tool
class OpticalFlowTrackingTool(BaseTool):
    """Tool for tracking features across video frames using optical flow."""
    
    name = "optical_flow_tracking"
    description = "Track features across multiple frames in a video using optical flow without requiring object detection."
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
            name="use_cuda",
            type=ToolParameterType.BOOLEAN,
            description="Whether to use CUDA for acceleration if available",
            required=False,
            default=True
        )
    ]
    
    @classmethod
    def execute(cls, start_time: float, end_time: float, 
               use_cuda: bool = False) -> Dict[str, Any]:
        """
        Track features across video frames using optical flow.
        
        Args:
            start_time: Start time in seconds for tracking
            end_time: End time in seconds for tracking
            use_cuda: Whether to use CUDA for acceleration
            
        Returns:
            Dictionary with tracking results
        """
        # Calculate duration
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
                "time_range": f"{requested_start_time:.1f}s to {requested_end_time:.1f}s"
            }
        
        # Calculate frame step to process approximately 2 frames per second (once per 0.5 seconds)
        # This reduces the computational load while maintaining tracking quality
        frame_step = max(1, int(fps / 2))
        
        logger.log_info(f"Tracking features using optical flow from frame {start_frame} to {end_frame} (processing every {frame_step} frames)")
        
        # Get first frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        ret, first_frame = cap.read()
        if not ret:
            logger.log_error(f"Could not read frame at position {start_frame}")
            return {
                "error": f"Could not read frame at position {start_frame}",
                "time_range": f"{requested_start_time:.1f}s to {requested_end_time:.1f}s"
            }
        
        # Initialize tracker
        logger.log_info(f"Initializing OpticalFlowTracker (CUDA: {use_cuda})")
        tracker = OpticalFlowTracker(use_cuda=use_cuda)
        tracker.init(first_frame)
        
        # Process frames
        all_tracks = []
        current_tracks = {}
        
        # Process frames
        frame_idx = start_frame
        frames_processed = 0
        
        # Calculate max frames to process (to show progress percentage)
        max_frames_to_process = (end_frame - start_frame) // frame_step + 1
        
        while frame_idx <= end_frame:
            # Set position and read frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
            
            # Update tracker
            updated_tracks = tracker.update(frame)
            frames_processed += 1
            
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
            
            # Update frame index - skip frames according to frame_step
            frame_idx += frame_step
            
            # Log progress
            if frames_processed % 10 == 0:  # Log every 10 processed frames
                progress_pct = min(100, int(frames_processed / max_frames_to_process * 100))
                logger.log_info(f"Progress: {progress_pct}% - Processed {frames_processed}/{max_frames_to_process} frames")
        
        # Calculate tracking statistics
        track_info = {}
        region_counts = {}
        
        for frame_tracks in all_tracks:
            for track in frame_tracks["tracks"]:
                track_id = track["id"]
                
                # Store trajectories and labels for each unique track
                if track_id not in track_info:
                    track_info[track_id] = {
                        "id": track_id,
                        "label": track["label"],
                        "first_seen": frame_tracks["time"],
                        "trajectory": []
                    }
                
                # Update last seen time
                track_info[track_id]["last_seen"] = frame_tracks["time"]
                
                # Add current position to trajectory (but not every frame to save space)
                if len(track_info[track_id]["trajectory"]) == 0 or \
                   frame_tracks["time"] - track_info[track_id]["trajectory"][-1]["time"] >= 0.5:  # Add point every 0.5s
                    track_info[track_id]["trajectory"].append({
                        "time": frame_tracks["time"],
                        "x": track["center"]["x"] / width,  # Normalize coordinates to 0-1 range
                        "y": track["center"]["y"] / height
                    })
                
                # Count features by region
                label = track["label"]
                if label not in region_counts:
                    region_counts[label] = 0
                region_counts[label] += 1
        
        # Find significant motion paths
        significant_paths = []
        
        for track_id, info in track_info.items():
            # Skip tracks with insufficient data
            if len(info.get("trajectory", [])) < 2:
                continue
                
            # Get start and end points (convert from normalized to pixel coordinates)
            trajectory = info["trajectory"]
            start_point = trajectory[0]
            end_point = trajectory[-1]
            
            start_x = int(start_point["x"] * width)
            start_y = int(start_point["y"] * height)
            end_x = int(end_point["x"] * width)
            end_y = int(end_point["y"] * height)
            
            # Calculate direction and distance
            dx = end_x - start_x
            dy = end_y - start_y
            distance = ((dx**2 + dy**2)**0.5)
            
            # Only include paths with significant movement
            if distance > width * 0.1 or distance > height * 0.1:
                # Determine direction
                if abs(dx) > abs(dy):
                    direction = "right" if dx > 0 else "left"
                else:
                    direction = "down" if dy > 0 else "up"
                
                significant_paths.append({
                    "region": info["label"],
                    "direction": direction,
                    "start_time": info["first_seen"],
                    "end_time": info["last_seen"],
                    "distance": int(distance)
                })
        
        # Create simplified summary
        summary = []
        
        # Add overall statistics
        total_features = len(track_info)
        if total_features > 0:
            summary.append(f"Tracked {total_features} features from {requested_start_time:.1f}s to {requested_end_time:.1f}s")
            
            # Group movements by direction
            direction_counts = {}
            for path in significant_paths:
                direction = path["direction"]
                if direction not in direction_counts:
                    direction_counts[direction] = 0
                direction_counts[direction] += 1
            
            # Report major motion directions
            if direction_counts:
                directions_desc = []
                for direction, count in sorted(direction_counts.items(), key=lambda x: x[1], reverse=True):
                    if count > 1:
                        directions_desc.append(f"{count} features moving {direction}")
                    else:
                        directions_desc.append(f"1 feature moving {direction}")
                
                if directions_desc:
                    summary.append("Detected motion: " + ", ".join(directions_desc[:3]))
            
            # Add active regions if there are any
            active_regions = []
            for label, count in sorted(region_counts.items(), key=lambda x: x[1], reverse=True):
                if count > 2:  # Only include regions with significant activity
                    active_regions.append(f"{label}")
            
            if active_regions and len(active_regions) <= 3:
                summary.append(f"Most active regions: {', '.join(active_regions[:3])}")
        else:
            summary.append(f"No significant motion detected from {requested_start_time:.1f}s to {requested_end_time:.1f}s")
        
        # Create the simplified result
        return {
            "time_range": f"{requested_start_time:.1f}s to {requested_end_time:.1f}s",
            "summary": summary,
            "significant_movements": len(significant_paths),
            "tracking_method": "optical_flow"
        } 