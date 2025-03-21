from typing import Dict, List, Any, Callable, Optional, Tuple
import traceback

from src.models.interfaces import Tool, Action, ActionStatus
from src.utils.logging import LoggingManager
from src.tools import TerminationSignal

logger = LoggingManager.get_logger()


class Executor:
    """
    Handles the execution of actions using registered tools.
    """
    
    def __init__(self):
        """Initialize the executor with an empty tool registry."""
        self.tools: Dict[str, Tool] = {}
    
    def register_tool(self, tool: Tool) -> None:
        """
        Register a tool for use by the executor.
        
        Args:
            tool: The tool to register
        """
        if tool.name in self.tools:
            logger.warning(f"Tool '{tool.name}' is already registered and will be overwritten.")
        
        self.tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")
    
    def register_tools(self, tools: List[Tool]) -> None:
        """
        Register multiple tools for use by the executor.
        
        Args:
            tools: List of tools to register
        """
        for tool in tools:
            self.register_tool(tool)
    
    def get_tool(self, tool_name: str) -> Optional[Tool]:
        """
        Get a registered tool by name.
        
        Args:
            tool_name: Name of the tool to retrieve
            
        Returns:
            The tool if found, None otherwise
        """
        return self.tools.get(tool_name)
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """
        List all registered tools in a serializable format.
        
        Returns:
            List of tool definitions
        """
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters
            }
            for tool in self.tools.values()
        ]
    
    def execute_action(self, action: Action) -> Tuple[bool, Any, Optional[str]]:
        """
        Execute a single action using a registered tool.
        
        Args:
            action: The action to execute
            
        Returns:
            Tuple containing:
            - Success flag (True if successful, False otherwise)
            - Result of the action (if successful)
            - Error message (if failed)
        """
        tool = self.get_tool(action.tool)
        
        if not tool:
            error_msg = f"Tool '{action.tool}' not found in registry"
            logger.error(error_msg)
            action.status = ActionStatus.FAILED
            action.error = error_msg
            return False, None, error_msg
        
        logger.info(f"Executing action with tool: {action.tool}")
        logger.debug(f"Action parameters: {action.params}")
        
        action.status = ActionStatus.IN_PROGRESS
        
        try:
            result = tool.function(**action.params)
            action.status = ActionStatus.COMPLETED
            action.result = result
            logger.info(f"Action completed successfully: {action.tool}")
            logger.info(f"Action result: {result}")
            return True, result, None
        except TerminationSignal as e:
            # Special handling for termination signal - let it propagate up
            # Set the action as completed with the termination result
            action.status = ActionStatus.COMPLETED
            action.result = {"reason": e.reason, "terminate": True}
            if hasattr(e, 'result') and e.result is not None:
                action.result["result"] = e.result
            logger.info(f"Termination signal raised by action: {action.tool}")
            # Re-raise to be caught by the orchestrator
            raise
        except Exception as e:
            error_msg = f"Error executing action with tool '{action.tool}': {str(e)}"
            stack_trace = traceback.format_exc()
            logger.error(f"{error_msg}\n{stack_trace}")
            
            action.status = ActionStatus.FAILED
            action.error = error_msg
            
            return False, None, error_msg
    
    def execute_actions(self, actions: List[Action]) -> List[Action]:
        """
        Execute a list of actions in sequence.
        
        Args:
            actions: List of actions to execute
            
        Returns:
            The updated list of actions with results and status
        """
        for action in actions:
            try:
                success, result, error = self.execute_action(action)
                
                # If the action is successful or has no fallbacks, continue to the next action
                if success or not action.fallbacks:
                    continue
                    
                # If we reach here, the action failed and has fallbacks
                logger.info(f"Attempting fallbacks for failed action: {action.tool}")
                
                # Sort fallbacks by priority (higher number = higher priority)
                fallbacks = sorted(action.fallbacks, key=lambda f: f.priority, reverse=True)
                
                for fallback in fallbacks:
                    logger.info(f"Trying fallback with tool: {fallback.tool}")
                    
                    fallback_action = Action(
                        tool=fallback.tool,
                        params=fallback.params,
                        purpose=f"Fallback for {action.tool}: {action.purpose}"
                    )
                    
                    try:
                        fallback_success, fallback_result, fallback_error = self.execute_action(fallback_action)
                        
                        if fallback_success:
                            logger.info(f"Fallback succeeded: {fallback.tool}")
                            action.status = ActionStatus.COMPLETED
                            action.result = fallback_result
                            break
                        else:
                            logger.warning(f"Fallback failed: {fallback.tool} - {fallback_error}")
                    except TerminationSignal:
                        # If a termination signal is raised during fallback execution, let it propagate
                        raise
            except TerminationSignal:
                # If a termination signal is raised, stop executing actions and let it propagate
                break
        
        return actions 