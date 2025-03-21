import asyncio
from agent.agent_coordinator import AgentCoordinator
from agent.question_evaluator import QuestionEvaluator
from agent.sub_question_generator import SubQuestionGenerator
from agent.question_solver import QuestionSolver
from agent.notebook_manager import NotebookManager
from tools.tool_manager import ToolManager
from models.deepseek_model import DeepseekModel

async def main():
    # Initialize components
    deepseek_model = DeepseekModel()
    tool_manager = ToolManager()
    notebook_manager = NotebookManager()
    
    question_evaluator = QuestionEvaluator(deepseek_model)
    sub_question_generator = SubQuestionGenerator(deepseek_model)
    question_solver = QuestionSolver(deepseek_model, tool_manager)
    
    agent_coordinator = AgentCoordinator(
        question_evaluator,
        sub_question_generator,
        question_solver,
        notebook_manager
    )
    
    # Example query
    query = "What emotions are expressed by the people in this cooking video?"
    context_knowledge = """
    The video shows a cooking show with two hosts. They are making a pasta dish.
    The video is 5 minutes long and includes close-ups of the preparation process.
    The hosts are explaining the steps as they cook.
    """
    
    # Process the query
    result = await agent_coordinator.process_query(query, context_knowledge)
    
    # Display the result
    print(f"Question: {query}")
    print(f"Answer: {result['answer']}")
    print(f"Entry ID: {result['entry_id']}")

if __name__ == "__main__":
    asyncio.run(main()) 