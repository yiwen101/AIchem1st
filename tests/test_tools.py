import unittest
from unittest.mock import patch, MagicMock
import os
import tempfile
import shutil

from src.tools.calculator import create_calculator_tool
from src.tools.query_llm import create_query_llm_tool
from src.tools.write_file import create_write_file_tool
from src.adapter.llm_adapter import LLMAdapter
from src.utils.config import Config


class TestCalculatorTool(unittest.TestCase):
    """Test cases for the calculator tool."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.calculator_tool = create_calculator_tool()
    
    def test_calculator_addition(self):
        """Test addition with the calculator tool."""
        result = self.calculator_tool.function(expression="2 + 2")
        self.assertEqual(result["result"], 4)
    
    def test_calculator_subtraction(self):
        """Test subtraction with the calculator tool."""
        result = self.calculator_tool.function(expression="10 - 5")
        self.assertEqual(result["result"], 5)
    
    def test_calculator_multiplication(self):
        """Test multiplication with the calculator tool."""
        result = self.calculator_tool.function(expression="3 * 4")
        self.assertEqual(result["result"], 12)
    
    def test_calculator_division(self):
        """Test division with the calculator tool."""
        result = self.calculator_tool.function(expression="20 / 4")
        self.assertEqual(result["result"], 5)
    
    def test_calculator_error(self):
        """Test error handling with the calculator tool."""
        result = self.calculator_tool.function(expression="10 / 0")
        self.assertIn("error", result)
        self.assertIn("division by zero", result["error"])


class TestQueryLLMTool(unittest.TestCase):
    """Test cases for the query LLM tool."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock the LLM adapter
        self.mock_adapter = MagicMock(spec=LLMAdapter)
        self.mock_adapter.generate.return_value = "This is a mock response from the LLM."
        
        # Create the tool with the mock adapter
        self.query_llm_tool = create_query_llm_tool(self.mock_adapter)
    
    def test_query_llm_basic(self):
        """Test basic querying with the LLM tool."""
        result = self.query_llm_tool.function(
            query="What is the capital of France?",
            system_prompt="You are a helpful assistant.",
            temperature=0.7,
            deep_thinking=True
        )
        
        # Verify the mock was called correctly
        self.mock_adapter.generate.assert_called_once()
        call_args = self.mock_adapter.generate.call_args[1]
        self.assertEqual(call_args["temperature"], 0.7)
        self.assertEqual(call_args["deep_thinking"], True)
        
        # Check the result
        self.assertIn("response", result)
        self.assertEqual(result["response"], "This is a mock response from the LLM.")
        self.assertIn("tokens", result)
    
    def test_query_llm_error_handling(self):
        """Test error handling with the LLM tool."""
        # Make the mock adapter raise an exception
        self.mock_adapter.generate.side_effect = Exception("API error")
        
        result = self.query_llm_tool.function(
            query="What is the meaning of life?",
            system_prompt="You are a helpful assistant."
        )
        
        # Check that the error was properly handled
        self.assertIn("error", result)
        self.assertEqual(result["error"], "API error")


class TestWriteFileTool(unittest.TestCase):
    """Test cases for the write to file tool."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.write_file_tool = create_write_file_tool()
        
        # Create a temporary output directory for testing
        self.original_output_dir = "output"
        self.test_output_dir = tempfile.mkdtemp()
        
        # Patch os.path.join to use our test directory
        self.join_patcher = patch('os.path.join')
        self.mock_join = self.join_patcher.start()
        self.mock_join.side_effect = lambda *args: os.path.join(self.test_output_dir, args[-1]) if args[0] == "output" else os.path.join(*args)
    
    def tearDown(self):
        """Clean up test fixtures."""
        # Remove temporary directory
        shutil.rmtree(self.test_output_dir)
        
        # Stop patchers
        self.join_patcher.stop()
        
    def test_write_file_basic(self):
        """Test basic file writing functionality."""
        result = self.write_file_tool.function(
            filename="test",
            content="This is a test file"
        )
        
        # Check the result
        self.assertTrue(result["success"])
        
        # Check that the file was created correctly
        filepath = os.path.join(self.test_output_dir, "test.md")
        self.assertTrue(os.path.exists(filepath))
        
        # Check file contents
        with open(filepath, 'r') as f:
            content = f.read()
            self.assertEqual(content, "This is a test file")
    
    def test_write_file_with_md_extension(self):
        """Test writing to a file with .md extension already specified."""
        result = self.write_file_tool.function(
            filename="test.md",
            content="This file already has an extension"
        )
        
        # Check the result
        self.assertTrue(result["success"])
        
        # Check that the file was created correctly
        filepath = os.path.join(self.test_output_dir, "test.md")
        self.assertTrue(os.path.exists(filepath))
        
        # Check file contents
        with open(filepath, 'r') as f:
            content = f.read()
            self.assertEqual(content, "This file already has an extension")
    
    def test_write_file_error_handling(self):
        """Test error handling when writing fails."""
        # Make write operation fail
        with patch('builtins.open') as mock_open:
            mock_open.side_effect = IOError("Permission denied")
            
            result = self.write_file_tool.function(
                filename="test_error",
                content="This should fail"
            )
            
            # Check the result
            self.assertFalse(result["success"])
            self.assertIn("error", result)
            self.assertIn("Permission denied", result["error"])


if __name__ == "__main__":
    unittest.main() 