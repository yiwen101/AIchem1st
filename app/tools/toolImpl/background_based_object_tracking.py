"""
Background-based object tracking using background subtraction.

This module provides a tracker that detects and tracks moving objects by
comparing each frame with a learned background model, without requiring
an external object detector.
"""

import os
import cv2
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import time
import matplotlib.pyplot as plt
import uuid

from app.tools.toolImpl.base_tool import BaseTool, ToolParameter, ToolParameterType
from app.tools.tool_manager import register_tool
from app.tools.resource.resource_manager import resource_manager
from app.common.monitor import logger


class BackgroundSubtractionTracker:
    """
    Tracker using background subtraction to detect and track moving objects.
    
    This tracker builds a model of the background and then identifies foreground
    objects as areas that differ significantly from the background.
    """
    
    def __init__(self, 
                 history: int = 500, 
                 threshold: float = 16, 
                 detect_shadows: bool = True,
                 learning_rate: float = 0.01,
                 min_area: int = 400):
        """
        Initialize the background subtraction tracker.
        
        Args:
            history: Length of history used by the background subtractor
            threshold: Threshold value for background subtraction
            detect_shadows: Whether to detect and mark shadows
            learning_rate: Rate at which background model is updated
            min_area: Minimum contour area to be considered an object
        """
        self.history = history
        self.threshold = threshold
        self.detect_shadows = detect_shadows
        self.learning_rate = learning_rate
        self.min_area = min_area
        
        # Initialize variables
        self.bg_subtractor = None
        self.tracks = {}
        self.next_id = 0
        self.frame_count = 0
        self.prev_frame = None
        self.frame_width = 0
        self.frame_height = 0
        
        # Kalman filter parameters for smoother tracking
        self.use_kalman = True
        self.kalman_filters = {}
        
        logger.log_info(f"Initialized BackgroundSubtractionTracker (min_area={min_area}, threshold={threshold})")
    
    def _init_bg_subtractor(self):
        """Initialize the background subtractor."""
        # MOG2 is generally better at handling lighting changes and shadows
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=int(self.history),
            varThreshold=float(self.threshold),
            detectShadows=bool(self.detect_shadows)
        )
        
        # Alternative background subtractors that could be used
        # self.bg_subtractor = cv2.createBackgroundSubtractorKNN(
        #     history=self.history,
        #     dist2Threshold=self.threshold,
        #     detectShadows=self.detect_shadows
        # )
    
    def _create_kalman_filter(self):
        """Create a Kalman filter for tracking."""
        # Kalman filter with 4 dynamic params (x, y, dx, dy) and 2 measurement params (x, y)
        kf = cv2.KalmanFilter(4, 2)
        
        # Measurement matrix (maps state vector to measurement vector)
        kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], np.float32)
        
        # Transition matrix (defines how the state evolves)
        kf.transitionMatrix = np.array([
            [1, 0, 1, 0],  # x = x + dx
            [0, 1, 0, 1],  # y = y + dy
            [0, 0, 1, 0],  # dx = dx
            [0, 0, 0, 1]   # dy = dy
        ], np.float32)
        
        # Process noise covariance
        kf.processNoiseCov = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], np.float32) * 0.03
        
        return kf
    
    def _assign_new_id(self) -> int:
        """Assign a new unique ID for a track."""
        track_id = self.next_id
        self.next_id += 1
        return track_id
    
    def _get_region_label(self, x: float, y: float) -> str:
        """Get a descriptive label for the region where the object is located."""
        # Vertical position
        if y < self.frame_height / 3:
            v_pos = "top"
        elif y < 2 * self.frame_height / 3:
            v_pos = "middle"
        else:
            v_pos = "bottom"
            
        # Horizontal position
        if x < self.frame_width / 3:
            h_pos = "left"
        elif x < 2 * self.frame_width / 3:
            h_pos = "center"
        else:
            h_pos = "right"
        
        return f"{h_pos}_{v_pos}_object"
    
    def init(self, frame: np.ndarray) -> None:
        """
        Initialize tracker with a frame.
        
        Args:
            frame: Initial video frame
        """
        # Reset state
        self.tracks = {}
        self.kalman_filters = {}
        self.frame_count = 0
        
        # Store frame dimensions
        self.frame_height, self.frame_width = frame.shape[:2]
        
        # Initialize background subtractor
        self._init_bg_subtractor()
        
        # Process the first frame (but don't detect objects yet)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.bg_subtractor.apply(gray, learningRate=1.0)  # Learn initial background
        
        # Store frame for later comparison
        self.prev_frame = gray
        
        logger.log_info(f"Background subtraction tracker initialized with frame size {self.frame_width}x{self.frame_height}")
    
    def _detect_objects(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect moving objects in a frame using background subtraction.
        
        Args:
            frame: Video frame
            
        Returns:
            List of detected objects
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(gray, learningRate=self.learning_rate)
        
        # Remove shadows (they are marked as gray (127))
        if self.detect_shadows:
            _, fg_mask = cv2.threshold(fg_mask, 127, 255, cv2.THRESH_BINARY)
        
        # Noise removal
        fg_mask = cv2.medianBlur(fg_mask, 5)
        
        # Morphology operations to fill holes and remove small noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Process each contour
        detections = []
        for contour in contours:
            # Filter small contours
            if cv2.contourArea(contour) < self.min_area:
                continue
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate center
            center_x = x + w // 2
            center_y = y + h // 2
            
            # Create detection
            detection = {
                "contour": contour,
                "box": {
                    "x": x,
                    "y": y,
                    "width": w,
                    "height": h
                },
                "center": {
                    "x": center_x,
                    "y": center_y
                },
                "area": cv2.contourArea(contour),
                "label": self._get_region_label(center_x, center_y)
            }
            
            detections.append(detection)
        
        return detections
    
    def _match_detections_to_tracks(self, detections: List[Dict]) -> List[Dict]:
        """
        Match new detections to existing tracks.
        
        Args:
            detections: List of detected objects
            
        Returns:
            List of matched tracks
        """
        # No existing tracks, create new ones for all detections
        if not self.tracks:
            for det in detections:
                track_id = self._assign_new_id()
                
                # Create new track
                self.tracks[track_id] = {
                    "id": track_id,
                    "label": det["label"],
                    "box": det["box"].copy(),
                    "center": det["center"].copy(),
                    "area": det["area"],
                    "last_seen": 0,
                    "age": 1,
                    "trajectory": [det["center"].copy()]
                }
                
                # Initialize Kalman filter for this track if enabled
                if self.use_kalman:
                    kf = self._create_kalman_filter()
                    
                    # Initialize state with current position and zero velocity
                    kf.statePost = np.array([
                        [det["center"]["x"]],
                        [det["center"]["y"]],
                        [0],  # initial dx
                        [0]   # initial dy
                    ], np.float32)
                    
                    self.kalman_filters[track_id] = kf
            
            return list(self.tracks.values())
        
        # We have existing tracks - match detections to tracks
        # Calculate cost matrix (distance between each detection and track)
        cost_matrix = np.zeros((len(self.tracks), len(detections)))
        
        for i, (track_id, track) in enumerate(self.tracks.items()):
            for j, det in enumerate(detections):
                # Calculate Euclidean distance between track and detection centers
                dx = track["center"]["x"] - det["center"]["x"]
                dy = track["center"]["y"] - det["center"]["y"]
                distance = np.sqrt(dx*dx + dy*dy)
                
                # Add size difference penalty
                track_area = track["area"]
                det_area = det["area"]
                area_ratio = max(track_area, det_area) / max(1, min(track_area, det_area))
                
                # Combined cost (distance + area difference)
                cost_matrix[i, j] = distance + (area_ratio - 1) * 50
        
        # Match detections to tracks greedily
        matched_tracks = set()
        matched_detections = set()
        
        # Maximum distance threshold for matching
        max_distance = min(self.frame_width, self.frame_height) * 0.2  # 20% of frame dimension
        
        # While there are possible matches
        while True:
            # Find lowest cost
            if cost_matrix.size == 0 or np.min(cost_matrix) > max_distance:
                break
                
            # Get indices of minimum cost
            i, j = np.unravel_index(np.argmin(cost_matrix), cost_matrix.shape)
            
            # Mark as matched
            track_id = list(self.tracks.keys())[i]
            matched_tracks.add(track_id)
            matched_detections.add(j)
            
            # Update track with detection
            det = detections[j]
            
            # Update with raw detection or Kalman filter
            if self.use_kalman:
                # Predict new state
                kf = self.kalman_filters[track_id]
                predicted_state = kf.predict()
                
                # Update with measurement
                measurement = np.array([[det["center"]["x"]], [det["center"]["y"]]], np.float32)
                corrected_state = kf.correct(measurement)
                
                # Get corrected position
                x_corrected = int(corrected_state[0, 0])
                y_corrected = int(corrected_state[1, 0])
                
                # Calculate velocity
                dx = corrected_state[2, 0]
                dy = corrected_state[3, 0]
                
                # Update center with Kalman-smoothed position
                self.tracks[track_id]["center"] = {
                    "x": x_corrected,
                    "y": y_corrected
                }
                
                # Update bounding box based on Kalman-smoothed center
                # Calculate new bounding box center
                center_x_offset = x_corrected - det["center"]["x"]
                center_y_offset = y_corrected - det["center"]["y"]
                
                # Adjust box coordinates
                self.tracks[track_id]["box"] = {
                    "x": det["box"]["x"] + center_x_offset,
                    "y": det["box"]["y"] + center_y_offset,
                    "width": det["box"]["width"],
                    "height": det["box"]["height"]
                }
            else:
                # Update directly with detection
                self.tracks[track_id]["box"] = det["box"].copy()
                self.tracks[track_id]["center"] = det["center"].copy()
            
            # Update other properties
            self.tracks[track_id]["label"] = det["label"]
            self.tracks[track_id]["area"] = det["area"]
            self.tracks[track_id]["last_seen"] = 0
            self.tracks[track_id]["age"] += 1
            self.tracks[track_id]["trajectory"].append(self.tracks[track_id]["center"].copy())
            
            # Set matched costs to infinity
            cost_matrix[i, :] = float('inf')
            cost_matrix[:, j] = float('inf')
        
        # Create new tracks for unmatched detections
        for j, det in enumerate(detections):
            if j not in matched_detections:
                track_id = self._assign_new_id()
                
                # Create new track
                self.tracks[track_id] = {
                    "id": track_id,
                    "label": det["label"],
                    "box": det["box"].copy(),
                    "center": det["center"].copy(),
                    "area": det["area"],
                    "last_seen": 0,
                    "age": 1,
                    "trajectory": [det["center"].copy()]
                }
                
                # Initialize Kalman filter for new track
                if self.use_kalman:
                    kf = self._create_kalman_filter()
                    
                    # Initialize state with current position and zero velocity
                    kf.statePost = np.array([
                        [det["center"]["x"]],
                        [det["center"]["y"]],
                        [0],
                        [0]
                    ], np.float32)
                    
                    self.kalman_filters[track_id] = kf
                
                matched_tracks.add(track_id)
        
        # Update last_seen for unmatched tracks
        for track_id in self.tracks:
            if track_id not in matched_tracks:
                self.tracks[track_id]["last_seen"] += 1
        
        # Remove tracks that haven't been seen for too long
        max_unseen_frames = 10  # Remove after 10 frames of not being seen
        tracks_to_remove = [track_id for track_id, track in self.tracks.items() 
                         if track["last_seen"] > max_unseen_frames]
        
        for track_id in tracks_to_remove:
            del self.tracks[track_id]
            if track_id in self.kalman_filters:
                del self.kalman_filters[track_id]
        
        # Return active tracks
        return [track for track_id, track in self.tracks.items() 
              if track["last_seen"] <= max_unseen_frames]
    
    def update(self, frame: np.ndarray) -> List[Dict]:
        """
        Update tracker with new frame.
        
        Args:
            frame: New video frame
            
        Returns:
            List of updated tracking results
        """
        # Increment frame count
        self.frame_count += 1
        
        # Detect objects in the current frame
        detections = self._detect_objects(frame)
        
        # Match detections with existing tracks
        updated_tracks = self._match_detections_to_tracks(detections)
        
        # Update previous frame
        self.prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        return updated_tracks


@register_tool
class BackgroundBasedTrackingTool(BaseTool):
    """Tool for tracking moving objects using background subtraction."""
    
    name = "background_based_tracking"
    description = "Track moving objects across frames using background subtraction without requiring an object detector."
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
            name="min_area",
            type=ToolParameterType.INTEGER,
            description="Minimum area (in pixels) for an object to be tracked",
            required=False,
            default=500
        ),
        ToolParameter(
            name="sensitivity",
            type=ToolParameterType.FLOAT,
            description="Sensitivity of motion detection (1-100, higher is more sensitive)",
            required=False,
            default=50.0
        )
    ]
    
    @classmethod
    def execute(cls, start_time: float, end_time: float, 
               min_area: int = 500, 
               sensitivity: float = 50.0) -> Dict[str, Any]:
        """
        Track moving objects across video frames using background subtraction.
        
        Args:
            start_time: Start time in seconds for tracking
            end_time: End time in seconds for tracking
            min_area: Minimum area (in pixels) for an object to be tracked
            sensitivity: Sensitivity of motion detection (1-100, higher is more sensitive)
            
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
        
        # Calculate frame step to process approximately 5 frames per second to balance
        # background model accuracy and processing speed
        frame_step = max(1, int(fps / 5))
        
        # Convert sensitivity to threshold (inverse relationship)
        # Higher sensitivity = lower threshold
        threshold = max(4, 36 - (sensitivity / 3))
        
        logger.log_info(f"Tracking objects using background subtraction from {start_time:.2f}s to {end_time:.2f}s" +
                      f" (processing every {frame_step} frames, threshold={threshold:.1f}, min_area={min_area})")
        
        # Process enough frames to learn the background before the actual tracking
        # This is crucial for accurate background modeling
        bg_learning_frames = min(int(fps * 1.0), 30)  # Learn from up to 1 second or 30 frames
        bg_learning_start = max(0, start_frame - bg_learning_frames)
        
        # Initialize tracker and learn background
        tracker = BackgroundSubtractionTracker(
            history=int(fps*2),  # 2 seconds of history
            threshold=float(threshold),
            detect_shadows=True,
            min_area=int(min_area),
            learning_rate=0.01  # Slow learning rate for stability
        )
        
        # Load first frame to initialize dimensions
        cap.set(cv2.CAP_PROP_POS_FRAMES, bg_learning_start)
        ret, first_frame = cap.read()
        if not ret:
            logger.log_error(f"Could not read frame at position {bg_learning_start}")
            return {
                "error": f"Could not read frame at position {bg_learning_start}",
                "time_range": f"{requested_start_time:.1f}s to {requested_end_time:.1f}s"
            }
        
        # Initialize tracker
        logger.log_info(f"Initializing background model from frame {bg_learning_start} to {start_frame}")
        tracker.init(first_frame)
        
        # Learn background
        for frame_idx in range(bg_learning_start + 1, start_frame, max(1, frame_step // 2)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
            
            # Update background model without tracking
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            initial_learning_rate = min(1.0, 3.0 / (start_frame - bg_learning_start))
            _ = tracker.bg_subtractor.apply(gray, learningRate=initial_learning_rate)
        
        # Process frames for tracking
        all_tracks = []
        current_tracks = {}
        
        # Process frames with the specified step size
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
            tracks_for_frame = [track for track_id, track in current_tracks.items() 
                               if "last_seen" in track and track["last_seen"] <= 5]
            
            # Only record frames with tracks if there are any
            if tracks_for_frame:
                all_tracks.append({
                    "frame": frame_idx,
                    "time": frame_time,
                    "tracks": tracks_for_frame
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
                
                # Count objects by region
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
            if distance > width * 0.05 or distance > height * 0.05:  # 5% of frame dimension
                # Determine direction
                if abs(dx) > abs(dy):
                    direction = "right" if dx > 0 else "left"
                else:
                    direction = "down" if dy > 0 else "up"
                
                # Calculate speed in pixels per second
                duration = info["last_seen"] - info["first_seen"]
                speed = distance / max(0.1, duration)  # Avoid division by zero
                
                # Create a detailed path with all points
                detailed_path = []
                for point in trajectory:
                    detailed_path.append({
                        "time": point["time"],
                        "x": int(point["x"] * width),
                        "y": int(point["y"] * height)
                    })
                
                # Extract key points (start, middle, end)
                key_points = []
                if len(trajectory) >= 2:
                    key_points.append(detailed_path[0])  # Start point
                    
                    # Add middle points for longer trajectories
                    if len(trajectory) > 3:
                        mid_idx = len(trajectory) // 2
                        key_points.append(detailed_path[mid_idx])
                    
                    key_points.append(detailed_path[-1])  # End point
                
                # Create a human-readable description of the movement
                movement_desc = (
                    f"Object {track_id} moved {direction} from ({start_x}, {start_y}) to ({end_x}, {end_y}) "
                    f"between {info['first_seen']:.2f}s and {info['last_seen']:.2f}s, "
                    f"covering {int(distance)} pixels at {int(speed)} pixels/second"
                )
                
                significant_paths.append({
                    "region": info["label"],
                    "direction": direction,
                    "start_time": info["first_seen"],
                    "end_time": info["last_seen"],
                    "distance": int(distance),
                    "duration": info["last_seen"] - info["first_seen"],
                    "speed": int(speed),
                    "path": detailed_path,
                    "key_points": key_points,
                    "description": movement_desc,
                    "start_position": {"x": start_x, "y": start_y},
                    "end_position": {"x": end_x, "y": end_y}
                })
        
        # Create visualization of movement paths if there are any
        visualization_path = None
        if significant_paths and len(significant_paths) > 0:
            try:
                # Get a frame for visualization background
                middle_frame = start_frame + (end_frame - start_frame) // 2
                cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
                ret, vis_frame = cap.read()
                
                if ret:
                    # Create a semi-transparent overlay for visualization
                    overlay = vis_frame.copy()
                    
                    # Draw paths with different colors
                    colors = [
                        (0, 0, 255),    # Red
                        (0, 255, 0),    # Green
                        (255, 0, 0),    # Blue
                        (0, 255, 255),  # Yellow
                        (255, 0, 255),  # Magenta
                        (255, 255, 0),  # Cyan
                        (128, 0, 0),    # Dark red
                        (0, 128, 0),    # Dark green
                        (0, 0, 128),    # Dark blue
                        (128, 128, 0),  # Olive
                    ]
                    
                    # Draw movements
                    for i, path in enumerate(significant_paths[:10]):  # Limit to top 10 for visibility
                        color = colors[i % len(colors)]
                        
                        # Draw line representing the path
                        if len(path["path"]) >= 2:
                            for j in range(len(path["path"]) - 1):
                                pt1 = (path["path"][j]["x"], path["path"][j]["y"])
                                pt2 = (path["path"][j+1]["x"], path["path"][j+1]["y"])
                                cv2.line(overlay, pt1, pt2, color, 2)
                        
                        # Draw start point (circle)
                        cv2.circle(overlay, (path["start_position"]["x"], path["start_position"]["y"]), 
                                 5, color, -1)
                        
                        # Draw end point (square)
                        end_x, end_y = path["end_position"]["x"], path["end_position"]["y"]
                        cv2.rectangle(overlay, (end_x-5, end_y-5), (end_x+5, end_y+5), color, -1)
                        
                        # Add ID label
                        cv2.putText(overlay, f"Obj {i}", (end_x+10, end_y), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    # Blend the overlay with the original frame
                    alpha = 0.7
                    cv2.addWeighted(overlay, alpha, vis_frame, 1-alpha, 0, vis_frame)
                    
                    # Add legend at the top of the frame
                    legend_text = "Movement Trajectories: circles=start, squares=end"
                    cv2.putText(vis_frame, legend_text, (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Save visualization
                    os.makedirs(f"app/tools/output/background_tracking/{video_name}", exist_ok=True)
                    visualization_path = f"app/tools/output/background_tracking/{video_name}/movement_{int(start_time)}_{int(end_time)}.jpg"
                    cv2.imwrite(visualization_path, vis_frame)
                    
                    logger.log_info(f"Saved movement visualization to {visualization_path}")
            except Exception as e:
                logger.log_error(f"Error creating visualization: {str(e)}")
        
        # Create detailed summary
        summary = []
        
        # Add overall statistics
        total_objects = len(track_info)
        if total_objects > 0:
            summary.append(f"Detected {total_objects} moving objects from {requested_start_time:.1f}s to {requested_end_time:.1f}s")
            
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
                        directions_desc.append(f"{count} objects moving {direction}")
                    else:
                        directions_desc.append(f"1 object moving {direction}")
                
                if directions_desc:
                    summary.append("Detected motion: " + ", ".join(directions_desc[:3]))
            
            # Add active regions if there are any
            active_regions = []
            for label, count in sorted(region_counts.items(), key=lambda x: x[1], reverse=True):
                if count > 2:  # Only include regions with significant activity
                    region_name = label.replace("_", " ")
                    active_regions.append(f"{region_name}")
            
            if active_regions and len(active_regions) <= 3:
                summary.append(f"Most active regions: {', '.join(active_regions[:3])}")
                
            # Add detailed descriptions of the most significant movements
            if significant_paths:
                # Sort paths by distance (longest first)
                sorted_paths = sorted(significant_paths, key=lambda x: x["distance"], reverse=True)
                
                # Add detailed descriptions for the top movements
                summary.append("Most significant movements:")
                for i, path in enumerate(sorted_paths[:5]):  # Limit to top 5 for readability
                    summary.append(f"{i+1}. {path['description']}")
        else:
            summary.append(f"No significant motion detected from {requested_start_time:.1f}s to {requested_end_time:.1f}s")
        
        # Create the detailed result
        result = {
            "time_range": f"{requested_start_time:.1f}s to {requested_end_time:.1f}s",
            "summary": summary,
            "significant_movements": len(significant_paths),
            "tracking_method": "background_subtraction"
        }
        
        # Add significant paths if there are any
        if significant_paths:
            # Sort by distance, longest first
            significant_paths.sort(key=lambda x: x["distance"], reverse=True)
            
            # Add detailed movement data
            result["paths"] = significant_paths[:10]  # Limit to top 10 paths
            
            # Add path descriptions for easier understanding
            result["movement_descriptions"] = [path["description"] for path in significant_paths[:10]]
        
        # Add visualization path if available
        if visualization_path:
            result["visualization"] = visualization_path
            
        # Add direction statistics
        if direction_counts:
            result["direction_stats"] = direction_counts
        
        return result 