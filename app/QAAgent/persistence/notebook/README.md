# QA Notebook Manager

A utility for managing question-answer records in JSON files.

## Overview

The notebook manager allows for:
- Loading QA records from JSON files
- Saving new QA records
- Updating notebooks with new QA pairs
- Resetting notebooks
- Generating prompts from notebook content

## Usage

```python
from app.context.notebook.notebook_manager import (
    load_notebook,
    update_notebook,
    save_notebook,
    reset_notebook,
    generate_prompt,
    list_notebooks
)

# List available notebooks
notebooks = list_notebooks()

# Load a notebook
records = load_notebook("example")

# Add a new QA pair
update_notebook(
    "example",
    question="What happens in the video?",
    answer="A person is walking through a park.",
    reason="Visual analysis shows a human figure moving along a path with trees."
)

# Generate a prompt from notebook content
prompt = generate_prompt("example")

# Reset a notebook (clear all records)
reset_notebook("example")
```

## Data Structure

Each notebook is stored as a JSON file in the `data/` directory. The file format is an array of records, where each record has:

- `question`: The question asked
- `answer`: The answer provided (if any)
- `reason`: The reasoning behind the answer (if any)
- `timestamp`: When the record was created

## Testing

You can run the test script to see how everything works:

```bash
python -m app.context.notebook.test_notebook_manager
``` 