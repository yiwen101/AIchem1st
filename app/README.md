# Video Understanding Agent

A LangGraph-based agent for answering questions about videos by breaking down complex queries, using tools, and building on previous answers.

## Structure

### State

The agent maintains the following state:
- `video_filename`: Path to the video being analyzed
- `question_stack`: Stack of questions to be answered (main question and sub-questions)
- `qa_notebook`: Record of previously answered questions with reasoning
- `tool_results`: Results from tool executions

### Nodes

The agent uses the following nodes:
- `setup`: Initializes the state for a new query
- `try_answer_with_past_QA`: Attempts to answer using previous QA records
- `try_answer_with_reasoning`: Attempts to answer using reasoning
- `is_primitive_question`: Determines if a question needs decomposition
- `decide_tool_calls`: Decides which tools to call
- `execute_tool_calls`: Executes the required tools
- `decompose_to_sub_question`: Breaks down complex questions
- `answer_query`: Formulates the final answer

### Flow

1. The agent starts with a question about a video
2. It first tries to answer using existing QA records
3. If that fails, it tries reasoning
4. If reasoning is insufficient, it either:
   - Determines the tools needed (for primitive questions)
   - Breaks down the question (for complex questions)
5. The process repeats until all questions/sub-questions are answered
6. The final answer is synthesized

## Usage

```python
from app.main import run_video_agent

result = run_video_agent("path/to/video.mp4", "What happens in this video?")
print(result["current_answer"])
```

Or from the command line:

```bash
python -m app.main path/to/video.mp4 "What happens in this video?"
```

## Implementation Notes

This is a skeleton implementation. The actual node functions need to be implemented to:
1. Connect to video processing tools
2. Use LLMs for reasoning
3. Implement proper question decomposition logic
4. Handle tool call execution 