import unittest

from src.tools.terminate import create_terminate_tool, TerminationSignal


class TestTerminateTool(unittest.TestCase):
    """Test cases for the terminate tool."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.terminate_tool = create_terminate_tool()
    
    def test_terminate_raises_exception(self):
        """Test that the terminate tool raises a TerminationSignal."""
        with self.assertRaises(TerminationSignal) as context:
            self.terminate_tool.function(reason="Testing termination")
        
        # Check that the exception message contains the reason
        self.assertIn("Testing termination", str(context.exception))
    
    def test_terminate_tool_properties(self):
        """Test the properties of the terminate tool."""
        self.assertEqual(self.terminate_tool.name, "terminate")
        self.assertIn("parameters", dir(self.terminate_tool))
        self.assertIn("reason", self.terminate_tool.parameters)


if __name__ == "__main__":
    unittest.main() 