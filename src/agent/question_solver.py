class QuestionSolver:
    def __init__(self, deepseek_model, tool_manager):
        """
        Initialize the QuestionSolver.
        
        Args:
            deepseek_model: The DeepSeek model instance
            tool_manager: The ToolManager instance
        """
        self.model = deepseek_model
        self.tool_manager = tool_manager
    
    def execute_tool_requests(self, tool_requests):
        """
        Execute a list of tool requests.
        
        Args:
            tool_requests: List of tool request dictionaries
            
        Returns:
            dict: Results from tool executions
        """
        if not tool_requests or not isinstance(tool_requests, list) or len(tool_requests) == 0:
            return {}
            
        print(f"Executing tools: {[req.get('tool') for req in tool_requests if req.get('tool')]}")
        return self.tool_manager.execute_tools(tool_requests) 