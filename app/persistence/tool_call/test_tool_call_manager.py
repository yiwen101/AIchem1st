"""
Test script for the tool call manager.
"""

import sys
import os
from pprint import pprint

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from app.persistence.tool_call.tool_call_manager import (
    load_tool_calls,
    add_tool_call,
    save_tool_calls,
    reset_tool_calls,
    reset_tool,
    generate_tool_call_summary,
    list_tool_call_files
)


def test_list_files():
    """Test listing available tool call files."""
    print("Available tool call files:")
    files = list_tool_call_files()
    for file in files:
        print(f"- {file}")
    print()


def test_load_tool_calls():
    """Test loading tool calls from a file."""
    print("Loading example tool calls:")
    tool_calls = load_tool_calls("example")
    for tool_name, calls in tool_calls.items():
        print(f"Tool: {tool_name}")
        print(f"Number of calls: {len(calls)}")
        print("First call args:", calls[0]["tool_args"] if calls else "None")
        print("First call result:", calls[0]["tool_result"] if calls else "None")
        print()


def test_generate_summary():
    """Test generating a summary from tool calls."""
    print("Generated summary from example tool calls:")
    summary = generate_tool_call_summary("example")
    print(summary)


def test_add_tool_call():
    """Test adding a new tool call."""
    print("Adding a new tool call to test file...")
    add_tool_call(
        "test",
        tool_name="video_captioning",
        tool_args={"video_path": "test_video.mp4", "time_segment": [0, 30]},
        tool_result="A person is giving a presentation in front of a large screen showing charts about climate data."
    )
    
    print("Updated test tool calls:")
    tool_calls = load_tool_calls("test")
    pprint(tool_calls)
    print()


def test_reset_tool():
    """Test resetting a specific tool's calls."""
    # First add another tool call for a different tool
    add_tool_call(
        "test",
        tool_name="object_recognition",
        tool_args={"frame_path": "frame_0.jpg"},
        tool_result=["person", "whiteboard", "projector"]
    )
    
    print("Tool calls before reset:")
    tool_calls = load_tool_calls("test")
    pprint(tool_calls)
    print()
    
    print("Resetting video_captioning tool...")
    reset_tool("test", "video_captioning")
    
    print("Tool calls after reset:")
    tool_calls = load_tool_calls("test")
    pprint(tool_calls)
    print()


def test_reset_all():
    """Test resetting all tool calls."""
    print("Resetting all tool calls in test file...")
    reset_tool_calls("test")
    
    print("Tool calls after complete reset:")
    tool_calls = load_tool_calls("test")
    pprint(tool_calls)
    print()


def main():
    """Run all tests."""
    print("=== TOOL CALL MANAGER TESTS ===\n")
    
    test_list_files()
    test_load_tool_calls()
    test_generate_summary()
    test_add_tool_call()
    test_reset_tool()
    test_reset_all()
    
    print("All tests completed.")


if __name__ == "__main__":
    main() 