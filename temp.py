from app.model.structs import VisionModelRequest
from eval import load_development_set
from app.common.resource_manager.resource_manager import resource_manager
from app.common.llm.openai import query_vision_llm
import random
from typing import Tuple, Optional
from pydantic import BaseModel
from app.tools.toolImpl.image_captioning import ImageCaptioningTool
from app.tools.tool_manager import execute_tool

class TemporalResponse(BaseModel):
    start_frame_index: int
    end_frame_index: int
    reasoning: str

def naive_temporal_query(resource_manager, query: str) -> Tuple[float, float]:
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
- start_time: The start time in seconds when relevant content begins
- end_time: The end time in seconds when relevant content ends
- confidence: A number from 0 to 1 indicating your confidence in this time range
- reasoning: Brief explanation of why you selected this time range

If you cannot determine a relevant time range, set start_time=0 and end_time={video_duration}.
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
        start_time = (start_frame_index-1) * unit_time 
        end_time = (end_frame_index - 1) * unit_time
        
        return (start_time, end_time)
    except Exception as e:
        print(f"Error determining temporal segment: {e}")
        return (0, video_duration)  # Default to full video if analysis fails


# first, generate the raw break down of questions
rows = load_development_set()
# pick one random row
row = random.choice(rows)
resource_manager.load_video_from_query(row)
# Example 1: When did the person in the video start talking?
query = row.question
print(f"Query: {query}")
start_time, end_time = naive_temporal_query(resource_manager, query)
print(f"Relevant segment: {start_time:.2f}s to {end_time:.2f}s")




