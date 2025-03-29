"""
Test script to verify a single query works end-to-end.
"""

import os
from app.model.structs import ParquetFileRow
from app.graph import create_video_agent_graph

def main():
    """Process a single test query."""
    # Create test query
    test_query = ParquetFileRow(
        qid="test-query-001",
        video_id="dQw4w9WgXcQ",  # Test video
        question_type="Test Question",
        capability="System Test",
        question="What is the main action in this video?",
        duration="3.5",
        question_prompt="Please analyze the video and describe what is happening.",
        answer="",
        youtube_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    )
    
    # Remove existing result file if any
    if os.path.exists("result.csv"):
        os.remove("result.csv")
    
    # Create graph with max steps limit
    print(f"Creating graph with max_steps=5")
    graph = create_video_agent_graph(max_steps=5)
    
    # Process query
    print(f"Processing query: {test_query.question}")
    result = graph.invoke(input={
        "query": test_query,
        "qa_notebook": [],
        "tool_results": {},
        "question_stack": [test_query.question],
        "task_queue": [],
        "current_question_tool_results": {},
        "previous_QA": None,
        "prev_attempt_answer_response": None,
        "step_count": 0,
        "max_steps": 5
    })
    
    # Print final step count
    print(f"\nFinal step count: {result['step_count']}/{result['max_steps']}")
    
    # Check result file
    if os.path.exists("result.csv"):
        print("\nResult file content:")
        with open("result.csv", "r") as f:
            print(f.read())
    else:
        print("\nResult file was not created.")

if __name__ == "__main__":
    main() 