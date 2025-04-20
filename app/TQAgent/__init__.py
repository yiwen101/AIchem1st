"""
TQAgent (Temporal Query Agent) module implementing the IVideoAgent interface.

This module provides a TQAgent class that uses temporal query methods
to analyze videos frame-by-frame with a structured approach.
"""

import os
import atexit
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import json

from app.model.interface import IVideoAgent
from app.model.structs import ParquetFileRow, VisionModelRequest, QueryVisionLLMResponse
from app.common.resource_manager.resource_manager import resource_manager
from app.common.monitor import logger
from app.common.llm.openai import query_vision_llm, single_query_llm_structured


class TQAgent(IVideoAgent):
    """
    TQAgent implements the IVideoAgent interface using a structured temporal query approach.
    
    This agent analyzes videos through a series of targeted frame-based methods
    that identify relevant segments and apply specialized analysis techniques
    based on the question type.
    """
    
    def __init__(self, num_frames: int = 15, model: str = "gpt-4o-mini"):
        """
        Initialize the TQAgent.
        
        Args:
            num_frames: Default number of frames to extract from the video
            model: The OpenAI model to use for answering questions
        """
        self.num_frames = num_frames
        self.model = model
        self.execution_history = []  # Store execution history
        
        # Ensure output directories exist
        os.makedirs("videos", exist_ok=True)
        os.makedirs("app/tools/output/tq_agent", exist_ok=True)
        
        # Register cleanup function
        atexit.register(self._cleanup_resources)
    
    def _cleanup_resources(self):
        """Clean up video resources on exit."""
        logger.log_info("Cleaning up resources...")
        resource_manager.cleanup()
    
    def _preload_video(self, video_id: str) -> bool:
        """
        Preload a video into the resource manager.
        
        Args:
            video_id: ID of the video to load
            
        Returns:
            True if the video was loaded successfully, False otherwise
        """
        video_path = f"videos/{video_id}.mp4"
        
        # Check if video exists
        if not os.path.exists(video_path):
            logger.log_warning(f"Video file not found: {video_path}")
            return False
        
        try:
            # Load video into resource manager
            metadata = resource_manager.load_video(video_path)
            logger.log_info(f"Loaded video {video_id} - Duration: {metadata['duration']:.2f}s, Resolution: {metadata['width']}x{metadata['height']}")
            return True
        except Exception as e:
            logger.log_error(f"Error loading video {video_id}: {str(e)}")
            return False

    # ---- Temporal Query Methods ----
    
    def find_relevant_frames(self, query: str, 
                           frame_description: str = "A sequence of frames from the video") -> Tuple[float, float]:
        """
        Find the start and end times that are most relevant to the query.
        
        Args:
            query: The query to find relevant frames for
            frame_description: Description of the frames
            
        Returns:
            Tuple of (start_time, end_time) in seconds
        """
        # Get video metadata
        _, metadata = resource_manager.get_active_video()
        video_duration = metadata['duration']
        
        # Extract frames evenly across the video
        frames, time_points = resource_manager.extract_frames_between(
            num_frames=self.num_frames,
            start_time=0,
            end_time=video_duration,
            save_frames=True,
            tool_name="tq_agent_find_relevant"
        )
        
        if not frames:
            logger.log_error("Failed to extract frames for finding relevant segments")
            return 0, video_duration
        
        # Format the prompt exactly as described in the planning section
        prompt = f"{frame_description}. Based on these frames, identify the start and end frame numbers that best show: {query}."
        prompt += "\nPlease provide your response in JSON format with 'start_frame' and 'end_frame' fields (0-indexed)."
        
        request = VisionModelRequest(prompt, frames)
        try:
            response = query_vision_llm(request, model=self.model)
            logger.log_info(f"Find relevant frames response: {response.answer}")
            
            # Extract start and end frames from response
            try:
                # Try to parse as JSON first
                response_dict = json.loads(response.answer)
                start_frame = int(response_dict.get('start_frame', 0))
                end_frame = int(response_dict.get('end_frame', len(frames) - 1))
            except (json.JSONDecodeError, ValueError, KeyError, TypeError):
                # If JSON parsing fails, try to extract numbers directly from the text
                logger.log_warning(f"Failed to parse JSON from response: {response.answer}")
                import re
                numbers = re.findall(r'\d+', response.answer)
                if len(numbers) >= 2:
                    start_frame = int(numbers[0])
                    end_frame = int(numbers[1])
                else:
                    # Default to full range if parsing fails
                    start_frame = 0
                    end_frame = len(frames) - 1
            
            # Ensure valid indices
            start_frame = max(0, min(start_frame, len(frames) - 1))
            end_frame = max(start_frame, min(end_frame, len(frames) - 1))
            
            # Convert frame indices to timestamps
           
            start_time = time_points[start_frame]
            end_time = time_points[end_frame]
            
            logger.log_info(f"Found relevant time range: {start_time:.2f}s to {end_time:.2f}s")
            return start_time, end_time
        except Exception as e:
            logger.log_error(f"Error finding relevant frames: {str(e)}")
            return 0, video_duration
    
    def identify_key_frame(self, event_description: str,
                         frame_description: str = "A sequence of frames from the video",
                         time_range: Tuple[float, float] | None = None) -> float:
        """
        Identify the timestamp that best captures a specific event.
        
        Args:
            event_description: Description of the event to find
            frame_description: Description of the frames
            time_range: Optional tuple of (start_time, end_time) to limit the search
            
        Returns:
            Timestamp of the key moment in seconds
        """
        # Get video metadata
        _, metadata = resource_manager.get_active_video()
        video_duration = metadata['duration']
        
        # Set default time range if not provided
        if time_range is None:
            start_time, end_time = 0, video_duration
        else:
            start_time, end_time = time_range
            # Ensure valid time range
            start_time = max(0, min(start_time, video_duration))
            end_time = max(start_time, min(end_time, video_duration))
        
        # Extract frames from the specified time range
        num_frames = min(15, max(5, int((end_time - start_time) / video_duration * self.num_frames * 2)))
        frames, time_points = resource_manager.extract_frames_between(
            num_frames=num_frames,
            start_time=start_time,
            end_time=end_time,
            save_frames=True,
            tool_name="tq_agent_identify_key"
        )
        
        if not frames:
            logger.log_error("Failed to extract frames for identifying key frame")
            return (start_time + end_time) / 2  # Return the middle of the time range
        
        # Format the prompt exactly as described in the planning section
        prompt = f"{frame_description}. Which frame number best captures when {event_description} occurs?"
        prompt += "\nRespond with only the frame number (0-indexed)."
        
        request = VisionModelRequest(prompt, frames)
        try:
            response = query_vision_llm(request, model=self.model)
            logger.log_info(f"Identify key frame response: {response.answer}")
            
            # Extract frame number from response
            try:
                # Try to extract the frame number
                import re
                match = re.search(r'\b(\d+)\b', response.answer.strip())
                if match:
                    frame_number = int(match.group(1))
                else:
                    # Default to middle frame if no number found
                    frame_number = len(frames) // 2
                
                # Ensure valid index
                frame_number = max(0, min(frame_number, len(frames) - 1))
                
                # Convert frame index to timestamp
                timestamp = time_points[frame_number]
                
                logger.log_info(f"Identified key moment at: {timestamp:.2f}s (frame {frame_number})")
                return timestamp
            except ValueError:
                # If parsing fails, return middle of the time range
                logger.log_warning(f"Failed to parse frame number from response: {response.answer}")
                return (start_time + end_time) / 2
        except Exception as e:
            logger.log_error(f"Error identifying key frame: {str(e)}")
            return (start_time + end_time) / 2
    
    def analyze_frames(self, query: str, 
                     frame_description: str = "The frames show a scene from the video",
                     time_range: Tuple[float, float] | None = None) -> str:
        """
        Analyze frames to answer a specific query.
        
        Args:
            query: The question to answer
            frame_description: Description of the frames
            time_range: Optional tuple of (start_time, end_time) to analyze
            
        Returns:
            Analysis result as text
        """
        # Get video metadata
        _, metadata = resource_manager.get_active_video()
        video_duration = metadata['duration']
        
        # Set default time range if not provided
        if time_range is None:
            start_time, end_time = 0, video_duration
            range_text = "all frames"
        else:
            start_time, end_time = time_range
            # Ensure valid time range
            start_time = max(0, min(start_time, video_duration))
            end_time = max(start_time, min(end_time, video_duration))
            range_text = f"time range {start_time:.2f}s to {end_time:.2f}s"
        
        # Extract frames from the specified time range
        num_frames = min(15, max(5, int((end_time - start_time) / video_duration * self.num_frames * 2)))
        frames, _ = resource_manager.extract_frames_between(
            num_frames=num_frames,
            start_time=start_time,
            end_time=end_time,
            save_frames=True,
            tool_name="tq_agent_analyze"
        )
        
        if not frames:
            logger.log_error("Failed to extract frames for analysis")
            return f"Error: Could not analyze frames from {range_text}."
        
        # Format the prompt exactly as described in the planning section
        prompt = f"{frame_description} from {range_text}. Answer the following question: {query}"
        prompt += "\nProvide a clear, concise answer based on what you can observe in these frames."
        
        request = VisionModelRequest(prompt, frames)
        try:
            response = query_vision_llm(request, model=self.model)
            logger.log_info(f"Analyze frames response: {response.answer}")
            return response.answer.strip()
        except Exception as e:
            logger.log_error(f"Error analyzing frames: {str(e)}")
            return f"Error analyzing the frames: {str(e)}"
    
    def detect_entities(self, entity_types: List[str],
                      frame_description: str = "The frames show a scene from the video",
                      time_range: Tuple[float, float] | None = None) -> str:
        """
        Detect entities of specified types in the frames.
        
        Args:
            entity_types: List of entity types to detect
            frame_description: Description of the frames
            time_range: Optional tuple of (start_time, end_time) to analyze
            
        Returns:
            Description of detected entities
        """
        # Get video metadata
        _, metadata = resource_manager.get_active_video()
        video_duration = metadata['duration']
        
        # Set default time range if not provided
        if time_range is None:
            start_time, end_time = 0, video_duration
            range_text = "all frames"
        else:
            start_time, end_time = time_range
            # Ensure valid time range
            start_time = max(0, min(start_time, video_duration))
            end_time = max(start_time, min(end_time, video_duration))
            range_text = f"time range {start_time:.2f}s to {end_time:.2f}s"
        
        # Extract frames from the specified time range
        num_frames = min(15, max(5, int((end_time - start_time) / video_duration * self.num_frames * 2)))
        frames, _ = resource_manager.extract_frames_between(
            num_frames=num_frames,
            start_time=start_time,
            end_time=end_time,
            save_frames=True,
            tool_name="tq_agent_detect"
        )
        
        if not frames:
            logger.log_error("Failed to extract frames for entity detection")
            return f"Error: Could not detect entities from {range_text}."
        
        # Format entity types as a string
        if isinstance(entity_types, list):
            entity_types_str = ", ".join(entity_types)
        else:
            entity_types_str = str(entity_types)
        
        # Format the prompt exactly as described in the planning section
        prompt = f"{frame_description} from {range_text}. Identify all {entity_types_str}. For each entity, describe its attributes and relevant details."
        prompt += "\nRespond with a detailed list of all detected entities and their properties."
        
        request = VisionModelRequest(prompt, frames)
        try:
            response = query_vision_llm(request, model=self.model)
            logger.log_info(f"Detect entities response: {response.answer}")
            return response.answer.strip()
        except Exception as e:
            logger.log_error(f"Error detecting entities: {str(e)}")
            return f"Error detecting entities: {str(e)}"
    
    def analyze_entity_state(self, entity_query: str,
                           timestamp: float = None,
                           frame_description: str = "This frame shows a moment from the video") -> str:
        """
        Analyze the state of an entity at a specific time.
        
        Args:
            entity_query: Description of the entity to analyze
            timestamp: Specific time in seconds to analyze, or None for the middle of the video
            frame_description: Description of the frame
            
        Returns:
            Description of the entity's state
        """
        # Get video metadata
        _, metadata = resource_manager.get_active_video()
        video_duration = metadata['duration']
        
        # Set default timestamp if not provided
        if timestamp is None:
            timestamp = video_duration / 2
        else:
            # Ensure valid timestamp
            timestamp = max(0, min(timestamp, video_duration))
        
        # Extract a single frame at the specified timestamp
        frame = resource_manager.get_frame_at_time(timestamp)[0]
        
        if frame is None or frame.size == 0:
            logger.log_error(f"Failed to extract frame at timestamp {timestamp:.2f}s")
            return f"Error: Could not analyze entity at timestamp {timestamp:.2f}s."
        
        # Format the prompt exactly as described in the planning section
        prompt = f"{frame_description} at timestamp {timestamp:.2f}s. Describe the state (appearance, position, emotion, etc.) of {entity_query}."
        prompt += "\nProvide a detailed description focusing on the requested entity."
        
        request = VisionModelRequest(prompt, [frame])
        try:
            response = query_vision_llm(request, model=self.model)
            logger.log_info(f"Analyze entity state response: {response.answer}")
            return response.answer.strip()
        except Exception as e:
            logger.log_error(f"Error analyzing entity state: {str(e)}")
            return f"Error analyzing entity state: {str(e)}"
    
    def count_entities(self, entity_query: str,
                      frame_description: str = "These frames show scenes from the video",
                      time_range: Tuple[float, float] | None = None) -> int:
        """
        Count the number of entities in the frames.
        
        Args:
            entity_query: Description of the entities to count
            frame_description: Description of the frames
            time_range: Optional tuple of (start_time, end_time) to analyze
            
        Returns:
            Count of entities
        """
        # Get video metadata
        _, metadata = resource_manager.get_active_video()
        video_duration = metadata['duration']
        
        # Set default time range if not provided
        if time_range is None:
            start_time, end_time = 0, video_duration
            range_text = "all frames"
        else:
            start_time, end_time = time_range
            # Ensure valid time range
            start_time = max(0, min(start_time, video_duration))
            end_time = max(start_time, min(end_time, video_duration))
            range_text = f"time range {start_time:.2f}s to {end_time:.2f}s"
        
        # Extract frames from the specified time range
        num_frames = min(15, max(5, int((end_time - start_time) / video_duration * self.num_frames * 2)))
        frames, _ = resource_manager.extract_frames_between(
            num_frames=num_frames,
            start_time=start_time,
            end_time=end_time,
            save_frames=True,
            tool_name="tq_agent_count"
        )
        
        if not frames:
            logger.log_error("Failed to extract frames for counting entities")
            return 0
        
        # Format the prompt exactly as described in the planning section
        prompt = f"{frame_description} from {range_text}. Count how many distinct {entity_query} appear."
        prompt += "\nRespond with only the number."
        
        request = VisionModelRequest(prompt, frames)
        try:
            response = query_vision_llm(request, model=self.model)
            logger.log_info(f"Count entities response: {response.answer}")
            
            # Extract count from response
            try:
                # Try to extract the number
                import re
                match = re.search(r'\b(\d+)\b', response.answer.strip())
                if match:
                    count = int(match.group(1))
                else:
                    # Default to 0 if no number found
                    logger.log_warning(f"No number found in response: {response.answer}")
                    count = 0
                return count
            except ValueError:
                logger.log_warning(f"Failed to parse count from response: {response.answer}")
                return 0
        except Exception as e:
            logger.log_error(f"Error counting entities: {str(e)}")
            return 0
    
    # ---- Question Processing ----
    
    def _determine_method_sequence(self, question: str, execution_history: List[Dict] = None) -> List[Dict[str, Any]]:
        """
        Determine the appropriate method sequence for a given question using LLM.
        
        Args:
            question: The question to analyze
            execution_history: Optional history of previous executions
            
        Returns:
            List of dictionaries containing method names and parameters
        """
        from pydantic import BaseModel, Field
        from typing import List, Dict, Optional, Any
        
        class MethodParam(BaseModel):
            name: str = Field(..., description="Name of the parameter")
            value: str = Field(..., description="Specific value for this parameter based on the question")
            
        class MethodStep(BaseModel):
            method: str = Field(..., description="Name of the method to call")
            description: str = Field(..., description="Explanation of what this method will do for this specific question")
            params: List[MethodParam] = Field(..., description="Parameters to use with this method")
            result_name: Optional[str] = Field(None, description="Name to assign to the result of this method call, for use in later steps")
            
        class MethodSequence(BaseModel):
            steps: List[MethodStep] = Field(..., description="Sequence of methods to call")
        
        # Construct the prompt with clear instructions and examples
        prompt = f"""
You are an expert video analysis system. Your task is to determine the most efficient sequence of methods to answer a question about a video. 

AVAILABLE METHODS:
1. find_relevant_frames(query, frame_description) -> frame_range: Finds relevant frames matching a query and returns a tuple (start_frame, end_frame). The returned frame_range can be used in subsequent methods that accept a frame_range parameter.
   - Best for: Finding segments where a specific event, person, or object appears
   - Example query: "the moment when the baby sees dad", "scenes with the man wearing a ring"
   - Formatted prompt: "[frame_description]. Based on these frames, identify the start and end frame numbers that best show: [query]."

2. identify_key_frame(event_description, frame_description) -> frame_number: Identifies a single frame that captures a specific event.
   - Best for: Pinpointing exact moments of specific actions or reactions
   - Example event_description: "when the baby smiles", "when the man takes off his hat"
   - Formatted prompt: "[frame_description]. Which frame number best captures when [event_description] occurs?"

3. analyze_frames(query, frame_description, frame_range) -> answer_text: Analyzes frames to answer a specific question. If frame_range is provided from a previous step, only those frames will be analyzed.
   - Best for: Direct questions about visible elements, actions, relationships, positions, or states
   - Example query: "Where is the ring worn by the man?", "What is the color of the car?", "How does the woman react?"
   - Formatted prompt: "[frame_description] from [range_text]. Answer the following question: [query]"
   - THIS IS THE MOST GENERAL METHOD and should be used as the final step for questions about locations, positions, attributes, emotions, etc.

4. detect_entities(entity_types, frame_description, frame_range) -> entity_list: Detects entities of specified types in frames.
   - Best for: Finding and describing objects, people, or elements in detail
   - Example entity_types: "people", "animals", "vehicles", "ring", "hat"
   - Formatted prompt: "[frame_description] from [range_text]. Identify all [entity_types_str]. For each entity, describe its attributes and relevant details."

5. analyze_entity_state(entity_query, timestamp, frame_description) -> entity_state: Analyzes state of an entity in a specific frame.
   - Best for: Detailed analysis of a specific object/person at a specific moment
   - Example entity_query: "the baby's face", "the man's emotional state", "the position of the ring"
   - Formatted prompt: "[frame_description] at timestamp [timestamp]. Describe the state (appearance, position, emotion, etc.) of [entity_query]."

6. count_entities(entity_query, frame_description, time_range) -> count: Counts instances of entities matching a description.
   - Best for: Questions asking "how many" of something appear
   - Example entity_query: "people", "dogs", "cars", "hats", "robot figures"
   - Formatted prompt: "[frame_description] from [range_text]. Count how many distinct [entity_query] appear."
   - ONLY use this for counting questions (e.g., "How many people are in the video?")

For the question: "{question}"
"""

        # If there's execution history, add it to the prompt
        if execution_history and len(execution_history) > 0:
            history_text = "\n\nPREVIOUS EXECUTION HISTORY:\n"
            for i, step in enumerate(execution_history):
                method = step.get("method", "unknown_method")
                params_str = ", ".join([f"{k}={v}" for k, v in step.get("params", {}).items()])
                result = step.get("result", "No result")
                result_name = step.get("result_name", "unnamed")
                history_text += f"{i+1}. {method}({params_str}) -> {result_name}: {result}\n"
            
            prompt += f"{history_text}\nBased on this previous execution and its results, create a better plan to answer the question."
        
        prompt += """
Select the most appropriate sequence of methods to efficiently answer this question. Be specific with parameter values, incorporating actual entities and descriptions from the question.

GUIDANCE FOR SPECIFIC QUESTION TYPES:
- For "WHERE" questions about position/location: Use analyze_frames as your final method, possibly after find_relevant_frames to narrow down to relevant sections
- For "HOW MANY" questions: Use count_entities as your final method
- For questions about emotional reactions: Consider using find_relevant_frames followed by analyze_frames
- For questions about specific moments: Use identify_key_frame followed by analyze_entity_state or analyze_frames

IMPORTANT RULES:
1. The result of the FINAL METHOD in your sequence will be used as the answer to the original question. Make sure your last method generates a meaningful answer.
2. Do NOT include methods that don't add value. Keep the sequence focused and minimal.
3. NAME each method's result with a descriptive name (e.g., 'key_frame', 'relevant_frames', 'entity_count').
4. When referencing a previous result, use exactly this format: "$result_name". The system will automatically replace "$result_name" with the actual value from the named result.

EXAMPLES:
- If find_relevant_frames returns a frame_range named 'action_frames', you would reference it as '$action_frames' in the frame_range parameter of analyze_frames.
- If identify_key_frame returns a frame_number named 'reaction_moment', you would reference it as '$reaction_moment' in the frame_number parameter of analyze_entity_state.

EXAMPLE PLANS:
1. "How many robot figures appear in the video?"
   Step 1: count_entities(entity_query="robot figures", frame_description="all frames from the video") → result_name="robot_count"

2. "Where is the ring worn by the man in the video?"
   Step 1: find_relevant_frames(query="man wearing a ring", frame_description="frames from the video") → result_name="relevant_frames"
   Step 2: analyze_frames(query="Where is the ring worn by the man?", frame_description="frames showing the man with a ring", frame_range="$relevant_frames") → result_name="ring_location"

3. "What is the baby's reaction when they see dad?"
   Step 1: find_relevant_frames(query="baby seeing dad", frame_description="scenes from the video") → result_name="meeting_moment"
   Step 2: analyze_frames(query="What is the baby's reaction when seeing dad?", frame_description="the moment when the baby sees dad", frame_range="$meeting_moment") → result_name="baby_reaction"
"""
        try:
            # Get structured output from LLM
            method_sequence = single_query_llm_structured(
                model=self.model,
                query=prompt,
                response_class=MethodSequence
            )
            
            # Convert the LLM response to our required format
            result = []
            for step in method_sequence.steps:
                # Create a parameters dictionary from the list of MethodParam objects
                params = {}
                for param in step.params:
                    params[param.name] = param.value
                
                step_dict = {
                    "method": step.method,
                    "params": params
                }
                
                # Add result_name if specified
                if step.result_name:
                    step_dict["result_name"] = step.result_name
                
                result.append(step_dict)
            
            logger.log_info(f"Method sequence determined: {result}")
            return result
            
        except Exception as e:
            logger.log_error(f"Error determining method sequence: {str(e)}")
            # Fallback to a simple default method if LLM call fails
            return []
            
    def _apply_method_sequence(self, method_sequence: List[Dict[str, Any]]) -> None:
        """
        Apply a sequence of methods to gather information about the video.
        This method executes the steps but doesn't formulate the final answer.
        
        Args:
            method_sequence: List of dictionaries with method names and parameters
        """
        if not method_sequence:
            logger.log_warning("No method sequence provided to apply")
            return
            
        # Clear previous execution history
        self.execution_history = []
        
        # Store results from each step
        step_results = {}
        
        # Get video metadata
        _, metadata = resource_manager.get_active_video()
        video_duration = metadata['duration']
        
        # Apply each method in sequence
        for i, step in enumerate(method_sequence):
            method_name = step["method"]
            params = step["params"].copy()  # Make a copy to avoid modifying the original
            
            # Check if any parameter values reference previous results (starting with $)
            for param_name, param_value in params.items():
                if isinstance(param_value, str) and param_value.startswith('$'):
                    # Extract the result name (remove the $ prefix)
                    result_name = param_value[1:]
                    
                    # Check if the referenced result exists
                    if result_name in step_results:
                        # Replace the reference with the actual value
                        params[param_name] = step_results[result_name]
                    else:
                        logger.log_warning(f"Referenced result '{result_name}' not found, using original parameter value")
            
            # Execution record to store in history
            execution_record = {
                "method": method_name,
                "params": params.copy(),
                "result_name": step.get("result_name", f"step_{i+1}_result")
            }
            
            # Dispatch to the appropriate method based on method_name
            try:
                logger.log_info(f"Applying method: {method_name} with params: {params}")
                result = None
                
                if method_name == "find_relevant_frames":
                    query = params.get("query", "")
                    frame_description = params.get("frame_description", "A sequence of frames from the video")
                    result = self.find_relevant_frames(query, frame_description)
                    logger.log_info(f"Found relevant time range: {result[0]:.2f}s to {result[1]:.2f}s")
                
                elif method_name == "identify_key_frame":
                    event_description = params.get("event_description", "")
                    frame_description = params.get("frame_description", "A sequence of frames from the video")
                    time_range = params.get("time_range", None)
                    result = self.identify_key_frame(event_description, frame_description, time_range)
                    logger.log_info(f"Identified key moment at: {result:.2f}s")
                
                elif method_name == "analyze_frames":
                    query = params.get("query", "")
                    frame_description = params.get("frame_description", "The frames show a scene from the video")
                    time_range = params.get("frame_range", None)  # Note: accepting 'frame_range' for backward compatibility
                    result = self.analyze_frames(query, frame_description, time_range)
                    logger.log_info(f"Analysis result: {result}")
                
                elif method_name == "detect_entities":
                    entity_types = params.get("entity_types", [])
                    if isinstance(entity_types, str):
                        entity_types = [entity_types]
                    frame_description = params.get("frame_description", "The frames show a scene from the video")
                    time_range = params.get("frame_range", None)  # Note: accepting 'frame_range' for backward compatibility
                    result = self.detect_entities(entity_types, frame_description, time_range)
                    logger.log_info(f"Detected entities: {result}")
                
                elif method_name == "analyze_entity_state":
                    entity_query = params.get("entity_query", "")
                    frame_description = params.get("frame_description", "This frame shows a moment from the video")
                    
                    # Handle both timestamp and frame_number for backward compatibility
                    timestamp = params.get("timestamp", None)
                    if timestamp is None:
                        frame_number = params.get("frame_number", None)
                        if frame_number is not None:
                            # Convert frame_number to timestamp if needed
                            timestamp = (frame_number / self.num_frames) * video_duration
                    
                    result = self.analyze_entity_state(entity_query, timestamp, frame_description)
                    logger.log_info(f"Entity state analysis: {result}")
                
                elif method_name == "count_entities":
                    entity_query = params.get("entity_query", "")
                    frame_description = params.get("frame_description", "These frames show scenes from the video")
                    time_range = params.get("frame_range", None)  # Note: accepting 'frame_range' for backward compatibility
                    result = self.count_entities(entity_query, frame_description, time_range)
                    logger.log_info(f"Entity count: {result}")
                
                else:
                    logger.log_warning(f"Unknown method: {method_name}")
                    result = None
                
                # Add result to execution record
                execution_record["result"] = result
                
                # Add execution record to history
                self.execution_history.append(execution_record)
                
                # Store the result with a name if provided
                if "result_name" in step and result is not None:
                    result_name = step["result_name"]
                    step_results[result_name] = result
                    logger.log_info(f"Stored result '{result}' with name '{result_name}'")
                
            except Exception as e:
                logger.log_error(f"Error applying method {method_name}: {str(e)}")
                execution_record["error"] = str(e)
                execution_record["result"] = None
                self.execution_history.append(execution_record)
            
    def _evaluate_answer(self, question: str, question_prompt: str) -> dict:
        """
        Evaluate if we can answer the question with the existing execution results.
        
        Args:
            question: The original question
            question_prompt: Additional prompt information for the question
            
        Returns:
            Dictionary with can_answer (bool) and answer (str) fields
        """
        from pydantic import BaseModel, Field
        from app.common.llm.openai import single_query_llm_structured
        
        class AnswerEvaluation(BaseModel):
            can_answer: bool
            answer: str
        
        if not self.execution_history:
            return {"can_answer": False, "answer": ""}
        
        # Create a prompt with the question and execution history
        prompt = f"""
You are a video analysis expert. Based on the question and the executed analysis, determine if you can answer the question.

QUESTION: {question}
ADDITIONAL PROMPT: {question_prompt}

EXECUTED ANALYSIS STEPS:
"""
        # Add each execution step to the prompt
        for i, step in enumerate(self.execution_history):
            method = step.get("method", "unknown_method")
            params_str = ", ".join([f"{k}={v}" for k, v in step.get("params", {}).items()])
            result = step.get("result", "No result")
            result_name = step.get("result_name", "unnamed")
            
            # Format the result based on its type
            if isinstance(result, tuple) and len(result) == 2:
                formatted_result = f"time range from {result[0]:.2f}s to {result[1]:.2f}s"
            elif isinstance(result, (int, float)):
                formatted_result = str(result)
            else:
                formatted_result = str(result)
                
            prompt += f"Step {i+1}: {method}({params_str}) → {result_name}: {formatted_result}\n"
        
        prompt += """
Based on these analysis steps and their results, can you answer the original question?

If YES:
1. Provide a clear, direct answer to the question
2. Base your answer solely on the information from the analysis steps
3. Synthesize information from multiple steps if needed

If NO:
1. Set can_answer to false
2. Leave the answer field empty

Your response should be structured as:
{
  "can_answer": true/false,
  "answer": "your answer here if can_answer is true, otherwise empty string"
}
"""
        try:
            # Query the LLM for structured evaluation
            response = single_query_llm_structured(
                model=self.model,
                query=prompt,
                response_class=AnswerEvaluation
            )
            
            logger.log_info(f"Answer evaluation: can_answer={response.can_answer}, answer={response.answer}")
            return {"can_answer": response.can_answer, "answer": response.answer}
        except Exception as e:
            logger.log_error(f"Error evaluating answer: {str(e)}")
            return {"can_answer": False, "answer": f"Could not evaluate answer: {str(e)}"}

    def get_answer(self, row: ParquetFileRow) -> str:
        """
        Get an answer to a question about a video.
        
        Args:
            row: ParquetFileRow containing the question and video information
            
        Returns:
            The answer to the question
        """
        # Load the video into resource manager
        logger.log_info(f"Processing query {row.qid} for video {row.video_id}")
        if not self._preload_video(row.video_id):
            logger.log_error(f"Could not load video for {row.qid}")
            return "I couldn't load the video to answer this question."
        
        try:
            # Set the number of frames to extract based on video duration
            duration = float(row.duration)
            self.num_frames = min(max(10, int(duration / 2)), 20)  # Between 10-20 frames depending on duration
            
            # First attempt - determine and apply method sequence
            logger.log_info(f"First attempt to answer question: {row.question}")
            method_sequence = self._determine_method_sequence(row.question)
            self._apply_method_sequence(method_sequence)
            
            # Evaluate if we can answer with the current execution history
            evaluation = self._evaluate_answer(row.question, row.question_prompt)
            
            # If we can answer, return the answer
            if evaluation["can_answer"]:
                logger.log_info(f"Successfully generated answer on first attempt")
                return evaluation["answer"]
                
            # If we get here, we need to try again with a revised plan
            logger.log_info(f"First attempt insufficient, trying again with execution history")
            method_sequence = self._determine_method_sequence(row.question, self.execution_history)
            
            if method_sequence:  # Only proceed if we got a valid sequence
                # Store the first execution history
                first_execution = self.execution_history.copy()
                
                # Apply the new method sequence
                self._apply_method_sequence(method_sequence)
                
                # Evaluate again
                second_evaluation = self._evaluate_answer(row.question, row.question_prompt)
                
                if second_evaluation["can_answer"]:
                    logger.log_info(f"Successfully generated answer on second attempt")
                    return second_evaluation["answer"]
                    
                # If we still can't answer, include first execution history in the evaluation
                logger.log_info(f"Second attempt still insufficient, combining execution histories")
                # Combine both execution histories for a final attempt
                self.execution_history = first_execution + self.execution_history
                final_evaluation = self._evaluate_answer(row.question, row.question_prompt)
                
                if final_evaluation["can_answer"]:
                    return final_evaluation["answer"]
            
            # If we're still here, we need to provide a best-effort answer
            logger.log_warning(f"Could not generate satisfactory answer after attempts")
            
            # Try to extract the most relevant result from execution history
            for step in reversed(self.execution_history):  # Start from the last step
                method = step.get("method")
                result = step.get("result")
                
                # Analyze_frames or count_entities are most likely to have direct answers
                if method in ["analyze_frames", "count_entities"] and result is not None:
                    if isinstance(result, str) and len(result.strip()) > 10:  # Non-empty string result
                        return f"{result}"
                    elif isinstance(result, (int, float)):
                        return f"{result}"
            
            # Fallback answer
            return "Based on the video analysis, I couldn't determine a clear answer to this question."
            
        except Exception as e:
            logger.log_error(f"Error processing query {row.qid}: {str(e)}")
            return f"Error processing the query: {str(e)}"
    
    def get_agent_name(self) -> str:
        """
        Get the name of this agent.
        
        Returns:
            The name of the agent
        """
        return f"TQAgent_{self.model}_{self.num_frames}frames"

# Export the TQAgent class
__all__ = ["TQAgent"] 