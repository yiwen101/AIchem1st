"""
Test script for the notebook manager.
"""

import sys
import os
from pprint import pprint

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from app.context.notebook.notebook_manager import (
    load_notebook,
    update_notebook,
    save_notebook,
    reset_notebook,
    generate_prompt,
    list_notebooks
)


def test_list_notebooks():
    """Test listing available notebooks."""
    print("Available notebooks:")
    notebooks = list_notebooks()
    for notebook in notebooks:
        print(f"- {notebook}")
    print()


def test_load_notebook():
    """Test loading a notebook."""
    print("Loading example notebook:")
    records = load_notebook("example")
    pprint(records)
    print()


def test_generate_prompt():
    """Test generating a prompt from a notebook."""
    print("Generated prompt from example notebook:")
    prompt = generate_prompt("example")
    print(prompt)
    print()


def test_update_notebook():
    """Test updating a notebook."""
    print("Updating test notebook with a new question-answer pair...")
    update_notebook(
        "test", 
        question="What is the weather like in the video?",
        answer="It appears to be a sunny day.",
        reason="Visual analysis shows bright lighting and shadows consistent with sunlight."
    )
    
    print("Updated test notebook:")
    records = load_notebook("test")
    pprint(records)
    print()


def test_reset_notebook():
    """Test resetting a notebook."""
    print("Resetting test notebook...")
    reset_notebook("test")
    
    print("Test notebook after reset:")
    records = load_notebook("test")
    pprint(records)
    print()


def main():
    """Run all tests."""
    print("=== NOTEBOOK MANAGER TESTS ===\n")
    
    test_list_notebooks()
    test_load_notebook()
    test_generate_prompt()
    test_update_notebook()
    test_reset_notebook()
    
    print("All tests completed.")


if __name__ == "__main__":
    main() 