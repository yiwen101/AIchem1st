from abc import ABC, abstractmethod
from typing import Dict, Any, List

class BaseTool(ABC):
    """Base class for all tools in the system."""
    
    @property
    def name(self) -> str:
        """Return the name of the tool."""
        return self.__class__.__name__
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Return a description of what the tool does."""
        pass
    
    @property
    @abstractmethod
    def input_schema(self) -> Dict[str, Dict[str, Any]]:
        """
        Return a schema describing the inputs of the tool.
        
        Returns:
            A dictionary with parameter names as keys and dictionaries with 'type',
            'description', and 'required' fields as values.
        """
        pass
    
    @property
    @abstractmethod
    def output_schema(self) -> Dict[str, Any]:
        """
        Return a schema describing the outputs of the tool.
        
        Returns:
            A dictionary describing the structure and types of the tool output.
        """
        pass
    
    @abstractmethod
    def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Execute the tool with the given parameters.
        
        Args:
            **kwargs: The parameters to pass to the tool
            
        Returns:
            A dictionary containing the results of the tool execution
        """
        pass
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get the complete metadata for this tool.
        
        Returns:
            A dictionary with name, description, input_schema, and output_schema
        """
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema
        } 