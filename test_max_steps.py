"""
Test script to verify max_steps functionality.
"""

from app.model.structs import ParquetFileRow
from app.graph import create_video_agent_graph


def main():
    """Test max_steps functionality."""
    # Create a sample query
    mock_query = ParquetFileRow(
        qid="test-001",
        video_id="dQw4w9WgXcQ",  # Rick roll video for testing
        question_type="Test Question",
        capability="Step Limit Test",
        question="What is happening in this video?",
        duration="3.33",
        question_prompt="Please analyze the video and describe what is happening.",
        answer="",
        youtube_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    )
    
    # Test with very low step limit to force termination
    print("Testing with max_steps=1 (should terminate quickly):")
    graph = create_video_agent_graph(max_steps=1)
    
    result = graph.invoke(input={
        "query": mock_query,
        "qa_notebook": [],
        "tool_results": {},
        "question_stack": [mock_query.question],
        "task_queue": [],
        "current_question_tool_results": {},
        "previous_QA": None,
        "prev_attempt_answer_response": None,
        "step_count": 0,
        "max_steps": 1
    })
    
    print("\nFinal state after test:")
    print(f"Step count: {result['step_count']}")
    print(f"Max steps: {result['max_steps']}")
    
    # Open and read result.csv to verify output
    try:
        with open("result.csv", "r") as f:
            lines = f.readlines()
            print("\nResults from result.csv:")
            for line in lines:
                print(line.strip())
    except FileNotFoundError:
        print("result.csv not found. The test may have failed.")


if __name__ == "__main__":
    main() 