"""
Tool manager for the video agent system.

This module provides a central manager for all tools used by the agent system.
It handles registration, documentation, and execution of tools.
"""

from typing import Dict, List, Any, Type, Optional, Callable
import inspect
from functools import wraps
import logging

from app.tools.base_tool import BaseTool, ToolParameter, ToolParameterType
from app.tools.tool_registry import register_tool, get_available_tools

logger = logging.getLogger(__name__)


class ToolManager:
    """
    Manager for all tools used by the agent system.
    
    This class provides methods to:
    1. Register tools
    2. Get information about all available tools
    3. Execute tools by name with parameters
    4. Validate parameters for tool execution
    """
    
    _instance = None
    
    def __new__(cls):
        """Create a singleton instance."""
        if cls._instance is None:
            cls._instance = super(ToolManager, cls).__new__(cls)
            cls._instance._tools = {}
            cls._instance._legacy_tools = {}
        return cls._instance
    
    def register_tool(self, tool_class: Type[BaseTool]) -> None:
        """
        Register a tool class.
        
        Args:
            tool_class: The tool class to register
            
        Returns:
            None
        """
        tool_name = tool_class.name
        if tool_name in self._tools:
            logger.warning(f"Tool {tool_name} already registered. Overwriting.")
        
        self._tools[tool_name] = tool_class
        
        # Register with tool registry for backward compatibility
        tool_class.register()
        
        logger.info(f"Registered tool: {tool_name}")
    
    def register_legacy_function(self, name: str, func: Callable, description: str = None) -> None:
        """
        Register a legacy function as a tool.
        
        Args:
            name: Name for the tool
            func: The function to register
            description: Optional description
            
        Returns:
            None
        """
        tool_class = BaseTool.from_function(func, name=name, description=description)
        self._legacy_tools[name] = tool_class
        self.register_tool(tool_class)
    
    def register_legacy_tools(self) -> None:
        """
        Register legacy tools from the tool registry.
        
        This method ensures backward compatibility with the existing tool registry.
        
        Returns:
            None
        """
        for name, func in get_available_tools().items():
            if name not in self._tools and name not in self._legacy_tools:
                self.register_legacy_function(name, func)
    
    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """
        Get schemas for all registered tools.
        
        Returns:
            List of tool schemas
        """
        return [tool.get_schema() for tool in self._tools.values()]
    
    def get_tool_schema(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """
        Get schema for a specific tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Tool schema or None if not found
        """
        if tool_name in self._tools:
            return self._tools[tool_name].get_schema()
        return None
    
    def execute_tool(self, tool_name: str, **kwargs) -> Any:
        """
        Execute a tool by name with parameters.
        
        Args:
            tool_name: Name of the tool to execute
            **kwargs: Parameters for the tool
            
        Returns:
            The result of the tool execution
            
        Raises:
            ValueError: If the tool is not found or parameters are invalid
        """
        if tool_name not in self._tools:
            raise ValueError(f"Tool '{tool_name}' not found. Available tools: {list(self._tools.keys())}")
        
        tool_class = self._tools[tool_name]
        
        # Validate parameters
        self._validate_parameters(tool_class, kwargs)
        
        # Execute the tool
        try:
            return tool_class.execute(**kwargs)
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {str(e)}")
            raise
    
    def _validate_parameters(self, tool_class: Type[BaseTool], params: Dict[str, Any]) -> None:
        """
        Validate parameters for a tool.
        
        Args:
            tool_class: The tool class
            params: The parameters to validate
            
        Raises:
            ValueError: If parameters are invalid
        """
        # Check for required parameters
        for param in tool_class.parameters:
            if param.required and param.name not in params:
                raise ValueError(f"Required parameter '{param.name}' missing for tool '{tool_class.name}'")
        
        # Add default values for missing optional parameters
        for param in tool_class.parameters:
            if not param.required and param.name not in params and param.default is not None:
                params[param.name] = param.default
    
    def format_tools_for_prompt(self) -> str:
        """
        Format tool information for inclusion in prompts.
        
        Returns:
            A formatted string with tool information
        """
        formatted = "# Available Tools\n\n"
        
        for tool_name, tool_class in sorted(self._tools.items()):
            formatted += f"## {tool_name}\n\n"
            formatted += f"{tool_class.description}\n\n"
            
            if tool_class.parameters:
                formatted += "Parameters:\n"
                for param in tool_class.parameters:
                    required_str = " (required)" if param.required else ""
                    default_str = f" (default: {param.default})" if not param.required and param.default is not None else ""
                    formatted += f"- `{param.name}` ({param.type.value}){required_str}{default_str}: {param.description}\n"
            else:
                formatted += "No parameters required.\n"
            
            formatted += "\n"
        
        return formatted


# Create singleton instance
tool_manager = ToolManager()


def register_cv_tool(tool_class: Type[BaseTool]) -> Type[BaseTool]:
    """
    Decorator to register a CV tool class.
    
    Args:
        tool_class: The tool class to register
        
    Returns:
        The registered tool class
    """
    tool_manager.register_tool(tool_class)
    return tool_class


def get_tool_schemas() -> List[Dict[str, Any]]:
    """
    Get schemas for all registered tools.
    
    Returns:
        List of tool schemas
    """
    return tool_manager.get_tool_schemas()


def execute_tool(tool_name: str, **kwargs) -> Any:
    """
    Execute a tool by name with parameters.
    
    Args:
        tool_name: Name of the tool to execute
        **kwargs: Parameters for the tool
        
    Returns:
        The result of the tool execution
        
    Raises:
        ValueError: If the tool is not found or parameters are invalid
    """
    return tool_manager.execute_tool(tool_name, **kwargs)


def format_tools_for_prompt() -> str:
    """
    Format tool information for inclusion in prompts.
    
    Returns:
        A formatted string with tool information
    """
    return tool_manager.format_tools_for_prompt() 