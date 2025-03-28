"""
Test script for the prompt helper.
"""

import sys
import os
from pprint import pprint

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from app.common.prompt.helper import (
    generate_prompt,
    format_notebook_info,
    format_tool_call_info,
    format_state_info,
    format_output_schema
)

from app.persistence.notebook.notebook_manager import update_notebook
from app.persistence.tool_call.tool_call_manager import add_tool_call


def test_format_state_info():
    """Test formatting state info."""
    print("=== Testing Format State Info ===")
    
    # Create a sample state
    state = {
        "video_filename": "example_video.mp4",
        "question_stack": ["What is happening in this video?", "How many people appear?"],
        "qa_notebook": [],
        "tool_results": {},
        "current_question_tool_results": {
            "video_analysis": "Found 3 people in the scene",
            "audio_transcription": "Person 1: Hello everyone, welcome to our presentation."
        },
        "previous_QA": {
            "question": "What is the background of the video?",
            "answer": "The background shows a city skyline with tall buildings.",
            "reason": "Visual analysis of the frames shows clear urban architecture."
        },
        "task_queue": [
            {"task": "Analyze audio", "status": "pending"},
            {"task": "Detect objects", "status": "completed"}
        ]
    }
    
    formatted = format_state_info(state)
    print(formatted)
    print()


def test_format_notebook_info():
    """Test formatting notebook info."""
    print("=== Testing Format Notebook Info ===")
    
    # Create a sample notebook
    update_notebook(
        "test_notebook", 
        "What is happening in the video?",
        "People are having a business meeting in a conference room.",
        "Visual analysis shows multiple people sitting around a table with presentation materials."
    )
    
    update_notebook(
        "test_notebook", 
        "How many people are in the video?",
        "There are 5 people in the video.",
        "Object detection identified 5 distinct human figures throughout the video."
    )
    
    # Sample notebook records (direct from state)
    notebook_records = [
        {
            "question": "What is happening in the video?",
            "answer": "People are having a business meeting in a conference room.",
            "reason": "Visual analysis shows multiple people sitting around a table with presentation materials."
        },
        {
            "question": "How many people are in the video?",
            "answer": "There are 5 people in the video.",
            "reason": "Object detection identified 5 distinct human figures throughout the video."
        }
    ]
    
    # Format directly from records (as would be in state)
    formatted = format_notebook_info(notebook_records)
    print(formatted)
    print()


def test_format_tool_call_info():
    """Test formatting tool call info."""
    print("=== Testing Format Tool Call Info ===")
    
    # Create a sample tool call file
    add_tool_call(
        "test_tool_calls",
        "video_analysis",
        {"video_path": "example_video.mp4", "analyze_type": "scene_detection"},
        "Found 3 scene changes at timestamps: 00:01:24, 00:03:45, 00:08:12"
    )
    
    add_tool_call(
        "test_tool_calls",
        "audio_transcription",
        {"video_path": "example_video.mp4", "segment": [0, 60]},
        "Hello everyone, welcome to our presentation on climate change solutions."
    )
    
    # Sample tool results (direct from state)
    tool_results = {
        "video_analysis": [
            "Found 3 scene changes at timestamps: 00:01:24, 00:03:45, 00:08:12"
        ],
        "audio_transcription": [
            "Hello everyone, welcome to our presentation on climate change solutions."
        ]
    }
    
    # Format directly from tool results (as would be in state)
    formatted = format_tool_call_info(tool_results)
    print(formatted)
    print()


def test_format_output_schema():
    """Test formatting output schema."""
    print("=== Testing Format Output Schema ===")
    
    schema = """
{
  "type": "object",
  "properties": {
    "answer": {
      "type": "string",
      "description": "The answer to the question"
    },
    "confidence": {
      "type": "number",
      "description": "Confidence score from 0 to 1"
    },
    "reasoning": {
      "type": "string",
      "description": "Reasoning behind the answer"
    }
  },
  "required": ["answer", "confidence", "reasoning"]
}
"""
    
    formatted = format_output_schema(schema)
    print(formatted)
    print()


def test_generate_prompt():
    """Test generating a complete prompt."""
    print("=== Testing Generate Prompt ===")
    
    # Create a sample state
    state = {
        "video_filename": "example_video.mp4",
        "question_stack": ["What is happening in this video?"],
        "qa_notebook": [
            {
                "question": "How many people are in the video?",
                "answer": "There are 5 people in the video.",
                "reason": "Object detection identified 5 distinct human figures."
            }
        ],
        "tool_results": {
            "video_analysis": ["Detected 5 people, 1 whiteboard, 3 laptops"],
            "audio_transcription": ["Person 1: Hello everyone, welcome to our presentation."]
        },
        "current_question_tool_results": {
            "scene_detection": "Office environment with natural lighting"
        }
    }
    
    schema = """
{
  "type": "object",
  "properties": {
    "answer": {
      "type": "string",
      "description": "The answer to the question"
    },
    "reasoning": {
      "type": "string",
      "description": "Reasoning behind the answer"
    }
  },
  "required": ["answer", "reasoning"]
}
"""
    
    # Generate a complete prompt
    prompt = generate_prompt(
        "What are the people in the video discussing?",
        state=state,
        notebook_info=True,
        tool_call_info=True,
        output_schema=schema
    )
    
    print(prompt)


def main():
    """Run all tests."""
    test_format_state_info()
    test_format_notebook_info()
    test_format_tool_call_info()
    test_format_output_schema()
    test_generate_prompt()


if __name__ == "__main__":
    main() 