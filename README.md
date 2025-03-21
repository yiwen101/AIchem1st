# Tool Planning Agent

A system for dynamic planning and execution of tool sequences while maintaining an evolving knowledge graph context.

## Overview

The Tool Planning Agent creates and executes plans to achieve user goals by:
- Breaking down goals into subgoal phases
- Dynamically generating actions for each subgoal on-the-fly
- Maintaining context in a knowledge graph
- Continuously evaluating progress and adapting plans
- Handling errors with fallback mechanisms

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd tool-planning-agent
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a configuration file (optional):
```json
{
  "api_key": "your-deepseek-api-key",
  "model": "deepseek-chat",
  "temperature": 0.7,
  "logging_level": "INFO"
}
```

## Usage

### Running from the command line

```bash
python -m src.main --goal "Find the weather in San Francisco and calculate the average temperature for the week" --config config.json
```

### Command line arguments

- `--goal`: The goal to achieve
- `--config`: Path to configuration file
- `--max-cycles`: Maximum number of execution cycles (default: 10)
- `--log-level`: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

### Using as a library

```python
from src.orchestrator.orchestrator import Orchestrator
from src.models.interfaces import Tool

# Initialize the orchestrator
orchestrator = Orchestrator("config.json")

# Register custom tools
def custom_tool(param1, param2):
    # Tool implementation
    return {"result": "some result"}

tool = Tool(
    name="custom_tool",
    description="Description of the tool",
    parameters={
        "param1": {"type": "string", "description": "Description of param1"},
        "param2": {"type": "integer", "description": "Description of param2"}
    },
    function=custom_tool
)
orchestrator.register_tool(tool)

# Initialize with a goal
orchestrator.initialize("Achieve something using custom tool")

# Execute the plan
result = orchestrator.execute_plan(max_cycles=5)
print(result)
```

## Available Tools

The Tool Planning Agent comes with the following default tools:

1. **Calculator Tool**: Evaluates mathematical expressions
   - Example: `2 + 2`, `5 * 10`, `(3 + 4) * 2`
   
2. **Query LLM Tool**: Interfaces with a thinking language model
   - Allows direct querying of the LLM with custom prompts
   - Supports system prompts and temperature adjustment
   - Enables deep thinking mode for more thorough responses
   
3. **Terminate Tool**: Ends the agent execution
   - Can be used to stop execution when the goal is achieved
   - Accepts a reason parameter explaining why execution is being terminated
   - Useful for preventing unnecessary actions when goal is achieved early

### Terminate Tool

The terminate tool allows agents to stop execution when the goal is achieved or no further action is necessary.

- **Name**: `terminate`
- **Parameters**:
  - `reason` (required): The reason for termination
  - `result` (optional): A dictionary containing the result to be written to the output file
- **Description**: Stops execution immediately and returns a status of "terminated" with the provided reason. If a result is provided, it will be written to the output file.

Example usage in a plan:
```json
{
  "name": "terminate",
  "parameters": {
    "reason": "Goal achieved: Successfully found the optimal solution.",
    "result": {
      "title": "Analysis Results",
      "summary": "The analysis is complete with the following findings...",
      "data": {...},
      "conclusion": "Based on these findings, we recommend..."
    }
  }
}
```

When called with a result, the tool will:
1. Create an output directory if it doesn't exist
2. Format the result as markdown
3. Write it to `output/output.md`
4. Raise a TerminationSignal exception that is caught by the orchestrator

## Creating Custom Tools

You can create custom tools by following the tool structure:

```python
from src.models.interfaces import Tool

def my_custom_function(param1, param2):
    # Implement your tool functionality
    return {"result": "Output from your tool"}

my_tool = Tool(
    name="my_custom_tool",
    description="Description of what your tool does",
    parameters={
        "param1": {"type": "string", "description": "Description of param1"},
        "param2": {"type": "integer", "description": "Description of param2"}
    },
    function=my_custom_function
)

# Register with the orchestrator
orchestrator.register_tool(my_tool)
```

## Architecture

The system consists of the following components:

1. **Orchestrator**: Coordinates the overall execution flow
2. **Planner**: Creates plans and generates actions dynamically
3. **Knowledge Manager**: Maintains the knowledge graph context
4. **Executor**: Registers and executes tools
5. **LLM Adapter**: Interfaces with the LLM for planning and reasoning

## Dynamic Workflow

The agent follows a dynamic workflow:

1. **Plan Creation**: Break down the goal into subgoal phases
   - Each phase represents a distinct step toward the main goal
   - No specific actions are predefined in the plan

2. **Action Generation**: For each phase, dynamically generate 1-3 actions
   - Actions are generated based on:
     - The current subgoal
     - Knowledge context
     - Previous action results
   - The system continuously adapts based on execution results

3. **Action Execution**: Execute the generated actions
   - Results are stored in both execution history and phase history
   - The knowledge graph is updated with new information

4. **Progress Evaluation**: After each action set:
   - Evaluate if the current phase/subgoal is complete
   - Either move to the next phase or generate more actions
   - Periodically revise the overall plan based on accumulated knowledge

5. **Plan Adaptation**: The system can:
   - Modify future phases based on newly discovered information
   - Change strategies if the current approach isn't working
   - Add new phases if unforeseen challenges arise

This approach allows for much more flexibility and adaptability compared to a rigid pre-planned sequence of actions.

## Future Enhancements

Future versions will add:
- Improved knowledge graph reasoning capabilities
- Enhanced context prioritization algorithms
- Collaborative multi-agent capabilities
- User feedback integration for plan adjustment