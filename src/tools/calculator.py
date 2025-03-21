"""
Calculator tool implementation for performing mathematical calculations.
"""

from typing import Dict, Any

from src.models.interfaces import Tool
from src.utils.logging import LoggingManager

logger = LoggingManager.get_logger()


def calculator(expression: str) -> Dict[str, Any]:
    """
    Evaluate a mathematical expression.
    
    Args:
        expression: The mathematical expression to evaluate
        
    Returns:
        Dictionary containing the result or an error message
    """
    logger.info(f"Calculating: {expression}")
    try:
        # Warning: eval is dangerous in production code
        # In a real implementation, use a safer alternative
        result = eval(expression)
        return {"result": result}
    except Exception as e:
        logger.error(f"Error in calculator: {str(e)}")
        return {"error": str(e)}


def create_calculator_tool() -> Tool:
    """
    Create a calculator tool instance.
    
    Returns:
        Tool instance for the calculator
    """
    return Tool(
        name="calculator",
        description="Evaluate mathematical expressions (e.g., '2 + 2', '5 * 10', etc.)",
        parameters={
            "expression": {
                "type": "string", 
                "description": "The mathematical expression to evaluate"
            }
        },
        function=calculator
    ) 