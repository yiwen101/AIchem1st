"""
Base Tool implementation for the video agent system.

This module provides a base class for all tools used in the agent system.
Tools inherit from this class to provide consistent interface and documentation.
"""

from typing import Dict, Any, List, Optional, Type, ClassVar, get_type_hints
from dataclasses import dataclass, field
import inspect
from enum import Enum
import json

from app.tools.tool_registry import register_tool


class ToolParameterType(Enum):
    """Enum for parameter types that can be used in tools."""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"
    
    @classmethod
    def from_python_type(cls, py_type):
        """Convert Python type to tool parameter type."""
        type_mapping = {
            str: cls.STRING,
            int: cls.INTEGER,
            float: cls.FLOAT,
            bool: cls.BOOLEAN,
            list: cls.ARRAY,
            dict: cls.OBJECT
        }
        return type_mapping.get(py_type, cls.STRING)


@dataclass
class ToolParameter:
    """Definition of a tool parameter."""
    name: str
    type: ToolParameterType
    description: str
    required: bool = True
    default: Any = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert parameter definition to dictionary."""
        result = {
            "name": self.name,
            "type": self.type.value,
            "description": self.description,
            "required": self.required
        }
        if self.default is not None:
            result["default"] = self.default
        return result


class BaseTool:
    """Base class for all tools in the agent system."""
    
    # Class variables that should be overridden by subclasses
    name: ClassVar[str] = "base_tool"
    description: ClassVar[str] = "Base tool class that all tools should inherit from."
    parameters: ClassVar[List[ToolParameter]] = []
    
    @classmethod
    def get_schema(cls) -> Dict[str, Any]:
        """Get the schema for this tool."""
        return {
            "name": cls.name,
            "description": cls.description,
            "parameters": [param.to_dict() for param in cls.parameters]
        }
    
    @classmethod
    def execute(cls, **kwargs) -> Any:
        """
        Execute the tool with the given parameters.
        
        Args:
            **kwargs: Parameters for the tool execution
            
        Returns:
            The result of the tool execution
        """
        raise NotImplementedError("Subclasses must implement execute method")
    
    @classmethod
    def register(cls) -> None:
        """Register this tool with the tool registry."""
        register_tool(cls.name)(cls.execute)
    
    @classmethod
    def from_function(cls, func, name=None, description=None):
        """
        Create a tool class from a function.
        
        Args:
            func: The function to convert to a tool
            name: Optional custom name for the tool
            description: Optional custom description for the tool
            
        Returns:
            A new tool class
        """
        func_name = name or func.__name__
        func_doc = description or func.__doc__ or f"Tool for {func_name}"
        
        # Inspect function signature to get parameters
        signature = inspect.signature(func)
        type_hints = get_type_hints(func)
        
        parameters = []
        for param_name, param in signature.parameters.items():
            # Skip self or cls parameters
            if param_name in ('self', 'cls'):
                continue
                
            param_type = type_hints.get(param_name, Any)
            default = param.default if param.default is not inspect.Parameter.empty else None
            required = param.default is inspect.Parameter.empty
            
            # Extract parameter description from docstring if available
            param_desc = f"Parameter {param_name} for {func_name}"
            if func.__doc__:
                # Simple docstring parser to extract parameter descriptions
                param_doc_match = func.__doc__.find(f"{param_name}:")
                if param_doc_match != -1:
                    param_doc_end = func.__doc__.find("\n", param_doc_match)
                    if param_doc_end != -1:
                        param_desc = func.__doc__[param_doc_match + len(param_name) + 1:param_doc_end].strip()
            
            parameters.append(ToolParameter(
                name=param_name,
                type=ToolParameterType.from_python_type(param_type),
                description=param_desc,
                required=required,
                default=default
            ))
        
        # Create new tool class
        tool_cls = type(
            f"{func_name.title()}Tool", 
            (cls,), 
            {
                "name": func_name,
                "description": func_doc,
                "parameters": parameters,
                "execute": staticmethod(func)
            }
        )
        
        return tool_cls 