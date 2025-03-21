"""
Test script to demonstrate the terminate tool with a result.
"""

import sys
import os
from src.tools import create_terminate_tool

# Make a nice test result
test_result = {
    "title": "Test Termination Result",
    "summary": "This is a test of the terminate tool with a result parameter",
    "data": [1, 2, 3, 4, 5],
    "conclusion": "The test was successful!"
}

# Create the terminate tool
terminate_tool = create_terminate_tool()

print("Testing terminate tool with result...")
try:
    # Call the terminate function with a result
    terminate_result = terminate_tool.function(
        reason="Testing termination with result",
        result=test_result
    )
    # This should not be reached
    print("ERROR: Terminate function did not raise an exception")
except Exception as e:
    print(f"Successfully caught exception: {type(e).__name__}")
    print(f"Reason: {str(e)}")
    print(f"Result written to output/output.md")
    
    # Print the content of the output file
    if os.path.exists("output/output.md"):
        print("\nOutput file content:")
        with open("output/output.md", "r") as f:
            print(f.read())
    else:
        print("ERROR: Output file was not created")

print("\nTest complete.") 