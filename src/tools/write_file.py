"""
Write to file tool implementation for writing content to files in the output directory.
"""

import os
from typing import Dict, Any

from src.models.interfaces import Tool
from src.utils.logging import LoggingManager

logger = LoggingManager.get_logger()


def write_to_file(filename: str, content: str) -> Dict[str, Any]:
    """
    Write content to a file in the output directory.
    
    Args:
        filename: The name of the file to write to (will be created in output/ directory)
        content: The content to write to the file
        
    Returns:
        Dictionary containing the result or an error message
    """
    logger.info(f"Writing to file: {filename}")
    
    # Ensure filename ends with .md
    if not filename.endswith(".md"):
        filename = f"{filename}.md"
    
    # Ensure output directory exists
    os.makedirs("output", exist_ok=True)
    
    # Construct full file path
    filepath = os.path.join("output", filename)
    
    try:
        with open(filepath, "w") as f:
            f.write(content)
        
        logger.info(f"Successfully wrote content to {filepath}")
        return {
            "success": True,
            "filepath": filepath,
            "message": f"Content written to {filepath}"
        }
    except Exception as e:
        error_msg = f"Error writing to file {filepath}: {str(e)}"
        logger.error(error_msg)
        return {
            "success": False,
            "error": error_msg
        }


def create_write_file_tool() -> Tool:
    """
    Create a write to file tool instance.
    
    Returns:
        Tool instance for writing to files
    """
    return Tool(
        name="write_to_file",
        description="Write content to a file in the output directory",
        parameters={
            "filename": {
                "type": "string", 
                "description": "The name of the file to write to (will be created in output/ directory, .md extension added by default)"
            },
            "content": {
                "type": "string",
                "description": "The content to write to the file"
            }
        },
        function=write_to_file
    ) 