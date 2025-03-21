import importlib
import inspect
from typing import Dict, List, Any
from tools.base_tool import BaseTool

class ToolManager:
    def __init__(self):
        """Initialize the tool manager with available tools."""
        self.tools = {}
        self._load_tools()
    
    def _load_tools(self):
        """Dynamically load all available tools."""
        tool_modules = [
            "ask_question_about_image",
            "video_frame_extractor" 
            # Add other tool modules as needed
            # "object_detector",
            # "action_recognizer",
            # "scene_classifier",
            # "transcript_generator", 
            # "facial_recognition",
            # "emotion_analyzer",
            # "temporal_analyzer"
        ]
        
        for module_name in tool_modules:
            try:
                # Import the module dynamically
                module = importlib.import_module(f"tools.{module_name}")
                
                # Find and register tool classes
                for name, obj in inspect.getmembers(module):
                    if (inspect.isclass(obj) and 
                        issubclass(obj, BaseTool) and 
                        obj is not BaseTool):
                        tool_instance = obj()
                        self.tools[tool_instance.name] = tool_instance
                        print(f"Registered tool: {tool_instance.name}")
            except Exception as e:
                print(f"Error loading tool module {module_name}: {str(e)}")
    
    def get_available_tools(self) -> List[str]:
        """Get a list of available tool names."""
        return list(self.tools.keys())
    
    def get_tool_metadata(self, tool_name: str = None) -> Dict[str, Any]:
        """
        Get metadata for a specific tool or all tools.
        
        Args:
            tool_name (str, optional): Name of the tool to get metadata for.
                                      If None, returns metadata for all tools.
        
        Returns:
            dict: Tool metadata
        """
        if tool_name:
            return self.tools.get(tool_name, {}).get_metadata() if tool_name in self.tools else {"error": f"Tool '{tool_name}' not found"}
        return {name: tool.get_metadata() for name, tool in self.tools.items()}
    
    def execute_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """
        Execute a specific tool with the provided parameters.
        
        Args:
            tool_name (str): Name of the tool to execute
            **kwargs: Parameters to pass to the tool
            
        Returns:
            dict: Results from the tool execution
        """
        # Check if tool exists
        if tool_name not in self.tools:
            return {
                "tool": tool_name,
                "success": False,
                "error": f"Tool '{tool_name}' not found",
                "available_tools": self.get_available_tools()
            }
        
        try:
            # Get the tool and validate required parameters
            tool = self.tools[tool_name]
            missing_params = [
                param for param, info in tool.input_schema.items() 
                if info.get("required", False) and param not in kwargs
            ]
            
            if missing_params:
                return {
                    "tool": tool_name,
                    "success": False,
                    "error": f"Missing required parameters: {', '.join(missing_params)}",
                    "required_params": {k: v for k, v in tool.input_schema.items() if v.get("required", False)}
                }
            
            # Execute the tool
            result = tool.execute(**kwargs)
            return {"tool": tool_name, "success": True, "result": result}
        except Exception as e:
            return {"tool": tool_name, "success": False, "error": str(e)}
    
    def execute_tools(self, tool_requests: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Execute multiple tools in sequence.
        
        Args:
            tool_requests (list): List of dicts with format:
            [
              {
                "tool": "ToolName",
                "params": {
                  "param1": "value1",
                  "param2": "value2"
                },
                "reason": "Explanation of why this tool is needed"
              }
            ]
            
        Returns:
            dict: Results from all tool executions
        """
        results = {}
        
        for request in tool_requests:
            tool_name = request.get("tool")
            params = request.get("params", {})
            
            if tool_name:
                # Execute the tool and store results
                results[tool_name] = self.execute_tool(tool_name, **params)
                
        return results 