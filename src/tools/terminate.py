"""
Terminate tool implementation for stopping agent execution.
"""

import os
from typing import Dict, Any, Optional

from src.models.interfaces import Tool
from src.utils.logging import LoggingManager

logger = LoggingManager.get_logger()


class TerminationSignal(Exception):
    """Exception raised to signal that execution should terminate."""
    def __init__(self, reason: str, result: Optional[Any] = None):
        self.reason = reason
        self.result = result
        super().__init__(reason)


def terminate(reason: str = "Goal accomplished", result: Optional[Any] = None) -> Dict[str, Any]:
    """
    Terminate the agent execution.
    
    Args:
        reason: The reason for termination
        result: Optional result data to save to output file
        
    Returns:
        A dictionary with the termination reason
        
    Raises:
        TerminationSignal: Always raised to signal termination
    """
    logger.info(f"Termination requested: {reason}")
    
    # Create output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)
    
    # If result is provided, write to output.md
    if result:
        try:
            with open("output/output.md", "w") as f:
                # If result is a string, write directly
                if isinstance(result, str):
                    f.write(result)
                # If result is a dict, format it as markdown
                elif isinstance(result, dict):
                    f.write("# Execution Result\n\n")
                    for key, value in result.items():
                        f.write(f"## {key}\n\n")
                        f.write(f"{value}\n\n")
                # Otherwise, convert to string
                else:
                    f.write(str(result))
            
            logger.info(f"Result written to output/output.md")
        except Exception as e:
            logger.error(f"Error writing result to output file: {str(e)}")
    
    response = {
        "reason": reason, 
        "terminate": True,
        "result_saved": result is not None
    }
    
    # Raise a termination signal to be caught by the orchestrator
    raise TerminationSignal(reason, result)


def create_terminate_tool() -> Tool:
    """
    Create a terminate tool instance.
    
    Returns:
        Tool instance for termination
    """
    return Tool(
        name="terminate",
        description="Terminate the agent execution when the goal is achieved or further execution is unnecessary",
        parameters={
            "reason": {
                "type": "string",
                "description": "The reason for termination (e.g., 'Goal accomplished', 'No further actions needed')"
            }
        },
        function=terminate
    ) 