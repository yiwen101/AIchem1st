"""
Terminate tool implementation for stopping agent execution.
"""

from typing import Dict, Any

from src.models.interfaces import Tool
from src.utils.logging import LoggingManager

logger = LoggingManager.get_logger()


class TerminationSignal(Exception):
    """Exception raised to signal that execution should terminate."""
    pass


def terminate(reason: str = "Goal accomplished") -> Dict[str, Any]:
    """
    Terminate the agent execution.
    
    Args:
        reason: The reason for termination
        
    Returns:
        A dictionary with the termination reason
        
    Raises:
        TerminationSignal: Always raised to signal termination
    """
    logger.info(f"Termination requested: {reason}")
    result = {"reason": reason, "terminate": True}
    
    # Raise a termination signal to be caught by the orchestrator
    raise TerminationSignal(reason)


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