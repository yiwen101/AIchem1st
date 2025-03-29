"""
Simple test script to verify that tools are working properly.
"""

from app.tools import tool_manager, format_tools_for_prompt

def main():
    """Test the tools functionality."""
    # Print available tools
    print("Available tools:")
    print(format_tools_for_prompt())
    
    # Print number of registered tools
    print(f"Total registered tools: {len(tool_manager._tools)}")
    
    # List registered tool names
    print("Registered tool names:")
    for name in sorted(tool_manager._tools.keys()):
        print(f"- {name}")

if __name__ == "__main__":
    main() 