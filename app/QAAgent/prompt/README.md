# Prompt Helper

A utility for generating formatted prompts with various contextual information.

## Overview

The prompt helper provides functions to:
- Generate complete prompts with state, notebook, and tool call information
- Format state information from the graph state
- Format QA records from the state's notebook
- Format tool call results from the state
- Add output schema information to prompts

## Usage

```python
from app.QAAgent.prompt import generate_prompt

# Basic usage with a simple prompt
prompt = generate_prompt("What is happening in this video?")

# Using with state and various information types
prompt = generate_prompt(
    "What is happening in this video?",
    state=state,                    # Current graph state
    notebook_info=True,             # Include notebook QA history from state
    tool_call_info=True,            # Include tool call history from state
    output_schema=json_schema_str   # Add schema requirements for output
)

# Using external files instead of state
prompt = generate_prompt(
    "What is happening in this video?",
    external_notebook_file="example",  
    external_tool_call_file="example"
)
```

## Advanced Usage

You can also use the individual formatting functions directly:

```python
from app.QAAgent.prompt.helper import (
    format_notebook_info,
    format_tool_call_info,
    format_state_info,
    format_output_schema
)

# Format state information for a prompt
state_info = format_state_info(state)

# Format notebook records directly from state
notebook_section = format_notebook_info(state["qa_notebook"])

# Format tool call results directly from state
tool_section = format_tool_call_info(state["tool_results"])

# Format output schema
schema_section = format_output_schema(json_schema_str)
```

## Testing

You can run the test script to see examples of how to use the prompt helper:

```bash
python -m app.QAAgent.prompt.test_helper
``` 