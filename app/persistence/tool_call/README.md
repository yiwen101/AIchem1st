# Tool Call Manager

A utility for managing and tracking tool calls in JSON files.

## Overview

The tool call manager allows for:
- Loading tool call information from JSON files
- Saving tool call results
- Adding new tool call records
- Resetting tool call history (for all tools or specific ones)
- Generating summaries of tool usage

## Usage

```python
from app.context.tool_call.tool_call_manager import (
    load_tool_calls,
    add_tool_call,
    save_tool_calls,
    reset_tool_calls,
    reset_tool,
    generate_tool_call_summary,
    list_tool_call_files
)

# List available tool call files
files = list_tool_call_files()

# Load tool calls from a file
tool_calls = load_tool_calls("example")

# Add a new tool call record
add_tool_call(
    "example",
    tool_name="video_analysis",
    tool_args={"video_path": "example.mp4", "analyze_type": "scene_detection"},
    tool_result="Found 5 scene changes"
)

# Generate a summary of tool calls
summary = generate_tool_call_summary("example")

# Reset a specific tool's calls
reset_tool("example", "video_analysis")

# Reset all tool calls
reset_tool_calls("example")
```

## Data Structure

Each tool call file is stored as a JSON file in the `data/` directory. The file format is a dictionary mapping tool names to arrays of tool call records, where each record has:

- `tool_name`: The name of the tool
- `tool_args`: The arguments passed to the tool
- `tool_result`: The result returned by the tool
- `timestamp`: When the tool call was made

## Testing

You can run the test script to see how everything works:

```bash
python -m app.context.tool_call.test_tool_call_manager
``` 