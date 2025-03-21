import importlib
import inspect

class ToolManager:
    def __init__(self):
        """Initialize the tool manager with available tools."""
        self.tools = {}
        self._load_tools()
    
    def _load_tools(self):
        """Dynamically load all available tools."""
        tool_modules = [
            "video_frame_extractor", 
            "object_detector",
            "action_recognizer",
            "scene_classifier",
            "transcript_generator", 
            "facial_recognition",
            "emotion_analyzer",
            "temporal_analyzer"
        ]
        
        for module_name in tool_modules:
            try:
                # Import the module dynamically
                module = importlib.import_module(f"..tools.{module_name}", package=__package__)
                
                # Find and register tool classes
                for name, obj in inspect.getmembers(module):
                    if (inspect.isclass(obj) and 
                        hasattr(obj, "execute") and 
                        callable(getattr(obj, "execute"))):
                        self.tools[name] = obj()
                        print(f"Registered tool: {name}")
            except Exception as e:
                print(f"Error loading tool module {module_name}: {str(e)}")
    
    def get_available_tools(self):
        """Get a list of available tools."""
        return list(self.tools.keys())
    
    def execute_tool(self, tool_name, **kwargs):
        """
        Execute a specific tool with the provided parameters.
        
        Args:
            tool_name (str): Name of the tool to execute
            **kwargs: Parameters to pass to the tool
            
        Returns:
            dict: Results from the tool execution
        """
        if tool_name not in self.tools:
            return {
                "error": f"Tool '{tool_name}' not found",
                "available_tools": self.get_available_tools()
            }
        
        try:
            result = self.tools[tool_name].execute(**kwargs)
            return {
                "tool": tool_name,
                "success": True,
                "result": result
            }
        except Exception as e:
            return {
                "tool": tool_name,
                "success": False,
                "error": str(e)
            }
    
    def execute_tools(self, tool_requests):
        """
        Execute multiple tools in sequence.
        
        Args:
            tool_requests (list): List of dicts with tool name and params
            
        Returns:
            dict: Results from all tool executions
        """
        results = {}
        
        for request in tool_requests:
            tool_name = request.get("tool")
            params = request.get("params", {})
            
            if tool_name:
                results[tool_name] = self.execute_tool(tool_name, **params)
        
        return results 