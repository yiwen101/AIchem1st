import os
import argparse
from typing import Dict, Any, List

from src.orchestrator.orchestrator import Orchestrator
from src.models.interfaces import Tool
from src.utils.config import Config
from src.utils.logging import LoggingManager
from src.tools import create_calculator_tool, create_query_llm_tool, create_terminate_tool

# Configure logger
logger = LoggingManager.get_logger()


def register_default_tools(orchestrator: Orchestrator) -> None:
    """
    Register a set of default tools with the orchestrator.
    
    Args:
        orchestrator: The orchestrator to register tools with
    """
    orchestrator.register_tools([
        create_calculator_tool(),
        create_query_llm_tool(orchestrator.llm_adapter),
        create_terminate_tool()
    ])

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Tool Planning Agent")
    parser.add_argument("--config", type=str, help="Path to configuration file", default="config.json")
    parser.add_argument("--goal", type=str, help="Goal to achieve")
    parser.add_argument("--max-cycles", type=int, default=10, help="Maximum execution cycles")
    parser.add_argument("--log-level", type=str, default="INFO", 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                       help="Logging level")
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Set up logging
    LoggingManager.configure(level=args.log_level)
    
    # Create orchestrator
    orchestrator = Orchestrator(args.config)
    
    # Register default tools
    register_default_tools(orchestrator)
    
    # Get goal from arguments or prompt user
    goal = args.goal
    if not goal:
        goal = input("Enter the goal to achieve: ")
    
    # Initialize the agent
    orchestrator.initialize(goal)
    
    # Execute the plan with dynamic action generation
    logger.info(f"Executing plan for goal: {goal}")
    result = orchestrator.execute_plan(max_cycles=args.max_cycles)
    
    # Print summary
    logger.info("Execution complete")
    logger.info(f"Status: {result['status']}")
    logger.info(f"Message: {result['message']}")
    
    # Print execution history
    print("\nExecution History:")
    for i, action in enumerate(result["execution_history"]):
        print(f"{i+1}. {action['phase']} - {action['tool']} - {action['status']}")
        if action["status"] == "completed":
            print(f"   Result: {action['result']}")
        elif action["status"] == "failed":
            print(f"   Error: {action['error']}")
    
    return result


if __name__ == "__main__":
    main() 