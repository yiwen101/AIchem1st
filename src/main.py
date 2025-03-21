import asyncio
from agent.agent_coordinator import AgentCoordinator
from agent.question_solver import QuestionSolver
from agent.notebook_manager import NotebookManager
from tools.tool_manager import ToolManager
from models.deepseek_model import DeepseekModel

async def main():
    # Initialize components
    deepseek_model = DeepseekModel()
    tool_manager = ToolManager()
    notebook_manager = NotebookManager()
    
    # Print available tools and their metadata
    print("Available tools:")
    tools_metadata = tool_manager.get_tool_metadata()
    for tool_name, metadata in tools_metadata.items():
        print(f"  - {tool_name}: {metadata['description']}")
        print(f"    Inputs: {', '.join(metadata['input_schema'].keys())}")
    print()
    
    # Create agent components
    question_solver = QuestionSolver(deepseek_model, tool_manager)
    
    agent_coordinator = AgentCoordinator(
        question_solver,
        notebook_manager
    )
    
    # Example query with context knowledge
    query = "Why the man in black suit and the man grey shirt stand face to face to each other in this image?"
    context_knowledge = """
    the image path is temp.png
    """
    
    # Process the query using the unified approach
    print("Processing query with unified approach...")
    result = await agent_coordinator.process_query(query, context_knowledge)
    
    # Display the result
    print("\nResults:")
    print(f"Question: {query}")
    print(f"Answer: {result['answer']}")
    print(f"Entry ID: {result['entry_id']}")

if __name__ == "__main__":
    asyncio.run(main()) 