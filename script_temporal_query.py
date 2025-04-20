from app.model.structs import VisionModelRequest
from app.tools.toolImpl.scene_detection import detect_scenes
from eval import load_development_set
from app.common.resource_manager.resource_manager import ResourceManager, resource_manager
from app.common.llm.openai import query_vision_llm
import random
from typing import Callable, List, Tuple
from pydantic import BaseModel
import cv2
import matplotlib.pyplot as plt
import numpy as np
import math

"""
This is meant to be tempo

"""


class TemporalResponse(BaseModel):
    start_frame_index: int
    end_frame_index: int
    reasoning: str

# define the type for temporal query
TemporalQueryFunction = Callable[[ResourceManager, str], Tuple[float, float]]
GenerateSceneFunction = Callable[[ResourceManager], List[np.ndarray]]

def uniform_scene_generation(resource_manager: ResourceManager) -> List[np.ndarray]:
    """
    Generate a list of frames from the active video.
    """
    video_metadata = resource_manager.get_active_video()[1]
    video_duration = video_metadata['duration']
    num_frames = 10  # Sample 10 frames across the video
    frames = resource_manager.extract_frames_between(
        num_frames=num_frames, 
        start_time=0, 
        end_time=video_duration,
        save_frames=True
    )
    return frames


def _llm_based_temporal_query(resource_manager, query: str, generate_scenes: GenerateSceneFunction) -> Tuple[float, float]:
    """
    Identify a time segment in a video that is relevant to the given query.
    
    Args:
        resource_manager: The resource manager with loaded video
        query: The text query to find a relevant segment for
        
    Returns:
        A tuple of (start_time, end_time) in seconds
    """
    # Extract frames from the video to understand content
    video_metadata = resource_manager.get_active_video()[1]
    video_duration = video_metadata['duration']
    num_frames = 10  # Sample 10 frames across the video
    frames = resource_manager.extract_frames_between(
        num_frames=num_frames, 
        start_time=0, 
        end_time=video_duration,
        save_frames=True
    )
    
    # Use LLM to determine relevant time range based on query and frame descriptions
    prompt = f"""
You are analyzing a video to identify the time segment most relevant to a query.
Based on the frames and their descriptions at different timestamps, determine the start and end times
where the content is most relevant to the query: "{query}"

Video duration: {video_duration} seconds

Return a JSON with the following fields:
- start_frame_index: The start frame index when relevant content begins, 0-indexed
- end_frame_index: The end frame index when relevant content ends, 0-indexed
- reasoning: Brief explanation of why you selected this time range

If you cannot determine a relevant time range, set start_frame_index=0 and end_frame_index={num_frames-1}.
"""
    request = VisionModelRequest(
        query=prompt,
        images=frames,
        response_class=TemporalResponse
    )
    try:
        response = query_vision_llm(request)
        start_frame_index = response.start_frame_index
        end_frame_index = response.end_frame_index
        reasoning = response.reasoning
        print(f"Start frame index: {start_frame_index}")
        print(f"End frame index: {end_frame_index}")
        print(f"Reasoning: {reasoning}")

        unit_time = video_duration / num_frames
        
        # Ensure values are within valid range
        start_time = (start_frame_index) * unit_time 
        end_time = (end_frame_index) * unit_time
        
        return (start_time, end_time)
    except Exception as e:
        print(f"Error determining temporal segment: {e}")
        return (0, video_duration)  # Default to full video if analysis fails

llm_based_uniform_temporal_query: TemporalQueryFunction = lambda resource_manager, query: _llm_based_temporal_query(resource_manager, query, uniform_scene_generation)
llm_based_scene_temporal_query: TemporalQueryFunction = lambda resource_manager, query: _llm_based_temporal_query(resource_manager, query, detect_scenes)

def eval_temporal_query(resource_manager, query: str, temporal_query: TemporalQueryFunction) -> Tuple[float, float]:
    """
    Evaluate a temporal query and visualize the results with a grid of frames.
    
    Args:
        resource_manager: The resource manager with loaded video
        query: The text query to find a relevant segment for
        temporal_query: The temporal query function to evaluate
        
    Returns:
        A tuple of (start_time, end_time) in seconds
    """
    # Get the temporal segment
    start_time, end_time = temporal_query(resource_manager, query)
    
    # Get metadata and extract frames
    metadata = resource_manager.get_active_video()[1]
    video_duration = metadata['duration']
    fps = metadata['fps']
    
    # Calculate number of frames to display (max 25 for readability)
    max_frames = 25
    num_frames = min(max_frames, int(video_duration))
    
    # Extract frames evenly distributed throughout the video
    frames = resource_manager.extract_frames_between(
        num_frames=num_frames, 
        start_time=0, 
        end_time=video_duration,
        save_frames=False
    )
    
    # Calculate grid dimensions
    grid_size = math.ceil(math.sqrt(num_frames))
    rows = grid_size
    cols = grid_size
    
    # Create figure and axes grid
    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
    
    # Flatten axes if it's a 2D array
    if rows > 1 and cols > 1:
        axes = axes.flatten()
    elif rows == 1:
        axes = [axes]
    
    # Calculate which frames fall within the temporal segment
    time_per_frame = video_duration / num_frames
    frame_times = [i * time_per_frame for i in range(num_frames)]
    
    # Plot each frame
    for i, (frame, timestamp) in enumerate(zip(frames, frame_times)):
        if i >= len(axes):
            break
            
        # Convert frame from BGR to RGB for matplotlib
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Display the frame
        axes[i].imshow(frame_rgb)
        
        # Add border to frames within the temporal segment
        if start_time <= timestamp <= end_time:
            # Add a red border to the relevant frames
            axes[i].spines['bottom'].set_color('red')
            axes[i].spines['top'].set_color('red')
            axes[i].spines['left'].set_color('red')
            axes[i].spines['right'].set_color('red')
            axes[i].spines['bottom'].set_linewidth(5)
            axes[i].spines['top'].set_linewidth(5)
            axes[i].spines['left'].set_linewidth(5)
            axes[i].spines['right'].set_linewidth(5)
            title_color = 'red'
        else:
            title_color = 'black'
        
        # Add timestamp as title
        axes[i].set_title(f"{timestamp:.1f}s", color=title_color, fontweight='bold' if start_time <= timestamp <= end_time else 'normal')
        axes[i].axis('off')  # Hide axis ticks
    
    # Hide any unused axes
    for i in range(len(frames), len(axes)):
        axes[i].axis('off')
    
    # Add overall title with query and time segment
    plt.suptitle(f"Query: {query}\nRelevant segment: {start_time:.2f}s - {end_time:.2f}s", fontsize=16)
    
    # Add color legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='white', edgecolor='red', linewidth=2, label='Relevant Segment')
    ]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.95, 0.98))
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Make room for title
    plt.show()
    
    # Also display a more detailed view of just the relevant segment
    if start_time < end_time:
        # Extract more frames from just the relevant segment
        segment_frames = resource_manager.extract_frames_between(
            num_frames=min(10, int((end_time - start_time) * fps)),  # Up to 10 frames from segment
            start_time=start_time,
            end_time=end_time,
            save_frames=False
        )
        
        if segment_frames:
            # Create a figure for the relevant segment
            segment_fig, segment_axes = plt.subplots(1, len(segment_frames), figsize=(15, 5))
            
            # Handle the case with only one frame
            if len(segment_frames) == 1:
                segment_axes = [segment_axes]
                
            # Calculate timestamps for each segment frame
            segment_duration = end_time - start_time
            segment_timestamps = [start_time + i * (segment_duration / len(segment_frames)) 
                                for i in range(len(segment_frames))]
            
            # Plot each frame in the segment
            for i, (frame, timestamp) in enumerate(zip(segment_frames, segment_timestamps)):
                # Convert frame from BGR to RGB for matplotlib
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Display the frame
                segment_axes[i].imshow(frame_rgb)
                segment_axes[i].set_title(f"{timestamp:.2f}s")
                segment_axes[i].axis('off')  # Hide axis ticks
            
            plt.suptitle(f"Detailed view of relevant segment: {start_time:.2f}s - {end_time:.2f}s", fontsize=16)
            plt.tight_layout()
            plt.subplots_adjust(top=0.85)  # Make room for title
            plt.show()
    
    return start_time, end_time

# first, generate the raw break down of questions
rows = load_development_set()
# pick one random row
row = random.choice(rows)
resource_manager.load_video_from_query(row)
# Example 1: When did the person in the video start talking?
query = row.question
print(f"Query: {query}")
#start_time, end_time = naive_temporal_query(resource_manager, query)
start_time, end_time = eval_temporal_query(resource_manager, query, llm_based_uniform_temporal_query)
print(f"Relevant segment: {start_time:.2f}s to {end_time:.2f}s")

start_time, end_time = eval_temporal_query(resource_manager, query, llm_based_scene_temporal_query)
print(f"Relevant segment: {start_time:.2f}s to {end_time:.2f}s")




