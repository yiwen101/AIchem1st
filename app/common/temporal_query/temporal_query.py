from app.model.structs import VisionModelRequest
from app.tools.toolImpl.scene_detection import detect_scenes
from eval import load_development_set
from app.common.resource_manager.resource_manager import ResourceManager, resource_manager
from app.common.llm.openai import query_vision_llm
from typing import Callable, List, Tuple
from pydantic import BaseModel
import numpy as np

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
    frames, _ = resource_manager.extract_frames_between(
        num_frames=num_frames, 
        start_time=0, 
        end_time=video_duration,
        save_frames=True
    )
    return frames

