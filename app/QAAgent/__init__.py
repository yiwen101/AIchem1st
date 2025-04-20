"""
QAAgent module implementing the IVideoAgent interface.

This module provides a QAAgent class that uses a graph-based approach to
answer questions about videos.
"""

import os
import atexit
from typing import Optional, Dict, Any

from app.model.interface import IVideoAgent
from app.model.structs import ParquetFileRow
from app.QAAgent.graph import create_video_agent_graph
from app.common.resource_manager.resource_manager import resource_manager
from app.common.monitor import logger
from langchain_core.runnables import RunnableConfig

class QAAgent(IVideoAgent):
    """
    QAAgent implements the IVideoAgent interface using a graph-based approach.
    
    This agent loads a video, processes it with various tools, and answers
    questions about the video content using a modular graph of nodes.
    """
    
    def __init__(self, max_steps: int = 20):
        """
        Initialize the QAAgent with the specified max steps.
        
        Args:
            max_steps: Maximum number of steps before terminating graph execution
        """
        self.max_steps = max_steps
        self.graph = create_video_agent_graph(max_steps=max_steps)
        
        # Ensure output directories exist
        os.makedirs("videos", exist_ok=True)
        os.makedirs("app/tools/output/image_captioning", exist_ok=True)
        os.makedirs("app/tools/output/object_detection", exist_ok=True)
        os.makedirs("app/tools/output/scene_detection", exist_ok=True)
        
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
    
    def get_answer(self, row: ParquetFileRow) -> str:
        """
        Get an answer to a question about a video.
        
        Args:
            row: ParquetFileRow containing the question and video information
            
        Returns:
            The answer to the question
        """
        # Load the video
        logger.log_info(f"Processing query {row.qid} for video {row.video_id}")
        if not self._preload_video(row.video_id):
            logger.log_error(f"Could not load video for {row.qid}")
            return "I couldn't load the video to answer this question."
        
        # Prepare the input state
        input_state = {
            "query": row,
            "qa_notebook": [],
            "tool_results": {},
            "question_stack": [row.question],
            "task_queue": [],
            "current_question_tool_results": {},
            "previous_QA": None,
            "prev_attempt_answer_response": None,
            # Initialize step count tracking
            "step_count": 0,
            "max_steps": self.max_steps
        }
        
        # Execute the graph
        try:
            result = self.graph.invoke(
                input=input_state, 
                config={"recursion_limit": 100}
            )
            
            # Extract the answer from the result
            if result and "qa_notebook" in result and result["qa_notebook"]:
                # Get the most recent QA record
                latest_qa = result["qa_notebook"][-1]
                return latest_qa.answer
            else:
                return "I wasn't able to find an answer to this question."
                
        except Exception as e:
            logger.log_error(f"Error processing query {row.qid}: {str(e)}")
            return f"Error processing the query: {str(e)}"
    
    def get_agent_name(self) -> str:
        """
        Get the name of this agent.
        
        Returns:
            The name of the agent
        """
        return "QAAgent"

# Export the QAAgent class
__all__ = ["QAAgent"] 