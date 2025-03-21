import unittest
from src.executor.executor import Executor
from src.models.interfaces import Tool, Action, ActionStatus


class TestExecutor(unittest.TestCase):
    """Test cases for the Executor component."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.executor = Executor()
        
        # Define a simple test tool
        def add_numbers(a: int, b: int):
            return {"result": a + b}
        
        self.add_tool = Tool(
            name="add_numbers",
            description="Add two numbers together",
            parameters={
                "a": {"type": "integer", "description": "First number"},
                "b": {"type": "integer", "description": "Second number"}
            },
            function=add_numbers
        )
        
        # Define a failing tool
        def failing_tool():
            raise ValueError("This tool always fails")
        
        self.failing_tool = Tool(
            name="failing_tool",
            description="A tool that always fails",
            parameters={},
            function=failing_tool
        )
        
        # Register tools
        self.executor.register_tools([self.add_tool, self.failing_tool])
    
    def test_tool_registration(self):
        """Test that tools are properly registered."""
        tools = self.executor.list_tools()
        self.assertEqual(len(tools), 2)
        self.assertEqual(tools[0]["name"], "add_numbers")
        self.assertEqual(tools[1]["name"], "failing_tool")
    
    def test_execute_action_success(self):
        """Test successful action execution."""
        action = Action(
            tool="add_numbers",
            params={"a": 5, "b": 7},
            purpose="Add 5 and 7"
        )
        
        success, result, error = self.executor.execute_action(action)
        
        self.assertTrue(success)
        self.assertEqual(result["result"], 12)
        self.assertIsNone(error)
        self.assertEqual(action.status, ActionStatus.COMPLETED)
    
    def test_execute_action_failure(self):
        """Test action execution failure."""
        action = Action(
            tool="failing_tool",
            params={},
            purpose="Test failure"
        )
        
        success, result, error = self.executor.execute_action(action)
        
        self.assertFalse(success)
        self.assertIsNone(result)
        self.assertIsNotNone(error)
        self.assertEqual(action.status, ActionStatus.FAILED)
    
    def test_execute_action_nonexistent_tool(self):
        """Test execution with a non-existent tool."""
        action = Action(
            tool="nonexistent_tool",
            params={},
            purpose="Test non-existent tool"
        )
        
        success, result, error = self.executor.execute_action(action)
        
        self.assertFalse(success)
        self.assertIsNone(result)
        self.assertIn("not found in registry", error)
        self.assertEqual(action.status, ActionStatus.FAILED)
    
    def test_execute_actions_with_fallback(self):
        """Test execution with fallbacks."""
        # Create an action with a failing tool but with a fallback to add_numbers
        action = Action(
            tool="failing_tool",
            params={},
            purpose="Test fallback",
            fallbacks=[
                {
                    "tool": "add_numbers",
                    "params": {"a": 10, "b": 20},
                    "priority": 1
                }
            ]
        )
        
        # Execute the action
        executed_actions = self.executor.execute_actions([action])
        
        # The action should have failed but the fallback should have succeeded
        self.assertEqual(executed_actions[0].status, ActionStatus.COMPLETED)
        self.assertEqual(executed_actions[0].result["result"], 30)


if __name__ == "__main__":
    unittest.main() 