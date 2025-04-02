#!/usr/bin/env python
"""
Manual tool testing script for AIchem1st video understanding tools.

Modify the variables below to test different tools with different parameters.
"""

import os
import sys
import time
import atexit
from pprint import pprint

# Add the current directory to the path so we can import the app modules
sys.path.append(os.getcwd())

from app.tools.resource.resource_manager import resource_manager
from app.tools import execute_tool, get_tool_schemas
from app.common.monitor import logger

# ===================== MODIFY THESE VARIABLES FOR TESTING =====================
# Video file name (must be in the 'videos' directory)
VIDEO_FILENAME = "yQ2YZQhvc2c.mp4"  # Change this to your video filename
'''
TOOL_NAME = "object_detection"

TOOL_PARAMS = {
    "time_seconds": 0.7,  # Example parameter for object_detection
}

'''
# Tool to execute
TOOL_NAME = "optical_flow_tracking"  # Change this to the tool you want to test

# Parameters for the tool - modify as needed
TOOL_PARAMS = {
    "start_time": 0.6,  # Example parameter for object_detection
    "end_time": 3.0
}


# Set to True to list all available tools and their parameters
LIST_TOOLS = False
# ===========================================================================

def cleanup_resources():
    """Clean up resources on exit."""
    logger.log_info("Cleaning up resources...")
    resource_manager.cleanup()
    logger.log_info("Resource cleanup complete")

def list_available_tools():
    """List all available tools and their parameters."""
    schemas = get_tool_schemas()
    print("\n===== Available Tools =====")
    
    for schema in schemas:
        print(f"\n{schema['name']}: {schema['description']}")
        if schema['parameters']:
            print("  Parameters:")
            for param in schema['parameters']:
                required = " (required)" if param.get('required', False) else ""
                default = f" (default: {param['default']})" if 'default' in param else ""
                print(f"    - {param['name']} ({param['type']}){required}{default}: {param['description']}")
    
    print("\n")

def print_result(result):
    """Pretty print the results with some formatting."""
    print("\n===== Tool Execution Result =====\n")
    
    # For special result types, provide additional formatting
    if isinstance(result, dict):
        # Handle image path outputs specially
        if 'output_path' in result and os.path.exists(result['output_path']):
            print(f"Output image saved to: {result['output_path']}")
            print(f"Full path: {os.path.abspath(result['output_path'])}")
        
        # Handle video output specially
        if 'output_video' in result and os.path.exists(result['output_video']):
            print(f"Output video saved to: {result['output_video']}")
            print(f"Full path: {os.path.abspath(result['output_video'])}")
    
    # Pretty print the full result
    pprint(result)
    print("\n")

def main():
    """Main function to execute the tool."""
    # Register cleanup function
    atexit.register(cleanup_resources)
    
    # Create output directories if they don't exist
    os.makedirs("videos", exist_ok=True)
    os.makedirs("app/tools/output/image_captioning", exist_ok=True)
    os.makedirs("app/tools/output/object_detection", exist_ok=True)
    os.makedirs("app/tools/output/scene_detection", exist_ok=True)
    os.makedirs("app/tools/output/object_tracking", exist_ok=True)
    
    # List tools if requested
    if LIST_TOOLS:
        list_available_tools()
        return
    
    # Load video
    video_path = f"videos/{VIDEO_FILENAME}"
    if not os.path.exists(video_path):
        logger.log_error(f"Video file not found: {video_path}")
        return
    
    logger.log_info(f"Loading video: {VIDEO_FILENAME}")
    try:
        # Load video into resource manager
        metadata = resource_manager.load_video(video_path)
        logger.log_info(f"Loaded video - Duration: {metadata['duration']:.2f}s, Resolution: {metadata['width']}x{metadata['height']}")
    except Exception as e:
        logger.log_error(f"Error loading video: {str(e)}")
        return
    
    # Execute tool
    logger.log_info(f"Executing tool: {TOOL_NAME} with parameters: {TOOL_PARAMS}")
    try:
        start_time = time.time()
        result = execute_tool(TOOL_NAME, **TOOL_PARAMS)
        end_time = time.time()
        
        logger.log_info(f"Tool execution completed in {end_time - start_time:.2f} seconds")
        
        # Print result
        print_result(result)
    except Exception as e:
        logger.log_exception(e, f"Error executing tool {TOOL_NAME}")
        print(f"\nError executing tool: {str(e)}")

if __name__ == "__main__":
    main() 