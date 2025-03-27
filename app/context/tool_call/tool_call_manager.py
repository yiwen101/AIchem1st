"""
Tool call manager for handling tool call information in JSON files.
"""

import os
import json
import time
from typing import Dict, List, Any, Optional, Union


class ToolCallInfo:
    """Information about a tool call."""
    def __init__(
        self, 
        tool_name: str, 
        tool_args: Dict[str, Any], 
        tool_result: Any,
        timestamp: Optional[float] = None
    ):
        self.tool_name = tool_name
        self.tool_args = tool_args
        self.tool_result = tool_result
        self.timestamp = timestamp or time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the tool call info to a dictionary."""
        return {
            "tool_name": self.tool_name,
            "tool_args": self.tool_args,
            "tool_result": self.tool_result,
            "timestamp": self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ToolCallInfo':
        """Create a tool call info from a dictionary."""
        return cls(
            tool_name=data["tool_name"],
            tool_args=data["tool_args"],
            tool_result=data["tool_result"],
            timestamp=data.get("timestamp", time.time())
        )


class ToolCallManager:
    """
    Manager for tool call information files.
    Handles loading, updating, saving, and reset operations.
    """
    
    BASE_DIR = "app/context/tool_call/data"
    
    def __init__(self):
        """Initialize the tool call manager."""
        # Ensure the data directory exists
        os.makedirs(self.BASE_DIR, exist_ok=True)
    
    def _get_file_path(self, filename: str) -> str:
        """
        Get the full path to a tool call file.
        
        Args:
            filename: The name of the tool call file (without .json extension)
            
        Returns:
            The full path to the tool call file
        """
        if not filename.endswith('.json'):
            filename = f"{filename}.json"
        
        return os.path.join(self.BASE_DIR, filename)
    
    def load(self, filename: str) -> Dict[str, List[ToolCallInfo]]:
        """
        Load tool call information from a file.
        
        Args:
            filename: The name of the tool call file
            
        Returns:
            A dictionary mapping tool names to lists of ToolCallInfo objects
        """
        file_path = self._get_file_path(filename)
        
        if not os.path.exists(file_path):
            return {}
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            result = {}
            for tool_name, tool_calls in data.items():
                result[tool_name] = [ToolCallInfo.from_dict(call) for call in tool_calls]
            
            return result
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error loading tool call info {filename}: {e}")
            return {}
    
    def save(self, filename: str, tool_calls: Dict[str, List[ToolCallInfo]]) -> bool:
        """
        Save tool call information to a file.
        
        Args:
            filename: The name of the tool call file
            tool_calls: Dictionary mapping tool names to lists of ToolCallInfo objects
            
        Returns:
            True if successful, False otherwise
        """
        file_path = self._get_file_path(filename)
        
        try:
            # Convert to serializable format
            data = {}
            for tool_name, calls in tool_calls.items():
                data[tool_name] = [call.to_dict() for call in calls]
            
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            return True
        except Exception as e:
            print(f"Error saving tool call info {filename}: {e}")
            return False
    
    def update(self, filename: str, tool_call: ToolCallInfo) -> bool:
        """
        Update a tool call file with a new tool call.
        
        Args:
            filename: The name of the tool call file
            tool_call: The new ToolCallInfo to add
            
        Returns:
            True if successful, False otherwise
        """
        tool_calls = self.load(filename)
        
        # Add the new tool call to the appropriate list
        if tool_call.tool_name not in tool_calls:
            tool_calls[tool_call.tool_name] = []
        
        tool_calls[tool_call.tool_name].append(tool_call)
        
        # Save the updated tool calls
        return self.save(filename, tool_calls)
    
    def add_tool_call(
        self, 
        filename: str, 
        tool_name: str, 
        tool_args: Dict[str, Any], 
        tool_result: Any
    ) -> bool:
        """
        Add a new tool call to the file.
        
        Args:
            filename: The name of the tool call file
            tool_name: The name of the tool
            tool_args: The arguments used for the tool call
            tool_result: The result of the tool call
            
        Returns:
            True if successful, False otherwise
        """
        tool_call = ToolCallInfo(
            tool_name=tool_name,
            tool_args=tool_args,
            tool_result=tool_result
        )
        return self.update(filename, tool_call)
    
    def reset(self, filename: str) -> bool:
        """
        Reset a tool call file (delete all tool calls).
        
        Args:
            filename: The name of the tool call file
            
        Returns:
            True if successful, False otherwise
        """
        return self.save(filename, {})
    
    def reset_tool(self, filename: str, tool_name: str) -> bool:
        """
        Reset a specific tool's calls in a file.
        
        Args:
            filename: The name of the tool call file
            tool_name: The name of the tool to reset
            
        Returns:
            True if successful, False otherwise
        """
        tool_calls = self.load(filename)
        
        if tool_name in tool_calls:
            tool_calls[tool_name] = []
            return self.save(filename, tool_calls)
        
        return True  # Nothing to do if the tool doesn't exist
    
    def list_files(self) -> List[str]:
        """
        List all available tool call files.
        
        Returns:
            A list of tool call filenames (without extension)
        """
        if not os.path.exists(self.BASE_DIR):
            return []
        
        files = [f for f in os.listdir(self.BASE_DIR) if f.endswith('.json')]
        return [f[:-5] for f in files]  # Remove .json extension
    
    def generate_tool_call_summary(self, filename: str) -> str:
        """
        Generate a summary of all tool calls from a file.
        
        Args:
            filename: The name of the tool call file
            
        Returns:
            A string summary of the tool calls
        """
        tool_calls = self.load(filename)
        
        if not tool_calls:
            return "No tool calls have been recorded."
        
        summary = "Tool Call Summary:\n\n"
        
        for tool_name, calls in tool_calls.items():
            summary += f"Tool: {tool_name}\n"
            summary += f"Number of calls: {len(calls)}\n"
            
            if calls:
                summary += "Recent calls:\n"
                # Sort by timestamp (newest first) and take the 5 most recent
                recent_calls = sorted(calls, key=lambda x: x.timestamp, reverse=True)[:5]
                
                for i, call in enumerate(recent_calls, 1):
                    summary += f"  Call {i}:\n"
                    summary += f"    Args: {json.dumps(call.tool_args, indent=2)}\n"
                    summary += f"    Result: {call.tool_result}\n"
                    summary += f"    Time: {time.ctime(call.timestamp)}\n"
            
            summary += "\n"
        
        return summary


# Create a singleton instance
tool_call_manager = ToolCallManager()


# Export functions for easy access

def load_tool_calls(filename: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Load tool call information from a file.
    
    Args:
        filename: The name of the tool call file (without .json extension)
        
    Returns:
        A dictionary mapping tool names to lists of tool call dictionaries
    """
    tool_calls = tool_call_manager.load(filename)
    return {tool_name: [call.to_dict() for call in calls] for tool_name, calls in tool_calls.items()}


def add_tool_call(
    filename: str, 
    tool_name: str, 
    tool_args: Dict[str, Any], 
    tool_result: Any
) -> bool:
    """
    Add a new tool call to the file.
    
    Args:
        filename: The name of the tool call file (without .json extension)
        tool_name: The name of the tool
        tool_args: The arguments used for the tool call
        tool_result: The result of the tool call
        
    Returns:
        True if successful, False otherwise
    """
    return tool_call_manager.add_tool_call(filename, tool_name, tool_args, tool_result)


def save_tool_calls(filename: str, tool_calls: Dict[str, List[Dict[str, Any]]]) -> bool:
    """
    Save tool call information to a file.
    
    Args:
        filename: The name of the tool call file (without .json extension)
        tool_calls: Dictionary mapping tool names to lists of tool call dictionaries
        
    Returns:
        True if successful, False otherwise
    """
    # Convert dictionaries to ToolCallInfo objects
    converted = {}
    for tool_name, calls in tool_calls.items():
        converted[tool_name] = [ToolCallInfo.from_dict(call) for call in calls]
    
    return tool_call_manager.save(filename, converted)


def reset_tool_calls(filename: str) -> bool:
    """
    Reset a tool call file (delete all tool calls).
    
    Args:
        filename: The name of the tool call file (without .json extension)
        
    Returns:
        True if successful, False otherwise
    """
    return tool_call_manager.reset(filename)


def reset_tool(filename: str, tool_name: str) -> bool:
    """
    Reset a specific tool's calls in a file.
    
    Args:
        filename: The name of the tool call file (without .json extension)
        tool_name: The name of the tool to reset
        
    Returns:
        True if successful, False otherwise
    """
    return tool_call_manager.reset_tool(filename, tool_name)


def generate_tool_call_summary(filename: str) -> str:
    """
    Generate a summary of all tool calls from a file.
    
    Args:
        filename: The name of the tool call file (without .json extension)
        
    Returns:
        A string summary of the tool calls
    """
    return tool_call_manager.generate_tool_call_summary(filename)


def list_tool_call_files() -> List[str]:
    """
    List all available tool call files.
    
    Returns:
        A list of tool call filenames (without extension)
    """
    return tool_call_manager.list_files() 