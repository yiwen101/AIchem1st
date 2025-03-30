"""
Notebook manager for handling QA records in JSON files.
"""

import os
import json
import time
from typing import Dict, List, Any, Optional, Union


class NotebookRecord:
    """A record in the QA notebook."""
    def __init__(self, question: str, answer: Optional[str] = None, reason: Optional[str] = None):
        self.question = question
        self.answer = answer
        self.reason = reason
        self.timestamp = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the record to a dictionary."""
        return {
            "question": self.question,
            "answer": self.answer,
            "reason": self.reason,
            "timestamp": self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NotebookRecord':
        """Create a record from a dictionary."""
        record = cls(
            question=data["question"],
            answer=data.get("answer"),
            reason=data.get("reason")
        )
        record.timestamp = data.get("timestamp", time.time())
        return record


class NotebookManager:
    """
    Manager for QA notebook files.
    Handles loading, updating, saving, and reset operations.
    """
    
    BASE_DIR = "app/context/notebook/data"
    
    def __init__(self):
        """Initialize the notebook manager."""
        # Ensure the data directory exists
        os.makedirs(self.BASE_DIR, exist_ok=True)
    
    def _get_file_path(self, filename: str) -> str:
        """
        Get the full path to a notebook file.
        
        Args:
            filename: The name of the notebook file (without .json extension)
            
        Returns:
            The full path to the notebook file
        """
        if not filename.endswith('.json'):
            filename = f"{filename}.json"
        
        return os.path.join(self.BASE_DIR, filename)
    
    def load(self, filename: str) -> List[NotebookRecord]:
        """
        Load a notebook from a file.
        
        Args:
            filename: The name of the notebook file
            
        Returns:
            A list of NotebookRecord objects
        """
        file_path = self._get_file_path(filename)
        
        if not os.path.exists(file_path):
            return []
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            records = [NotebookRecord.from_dict(record) for record in data]
            return records
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error loading notebook {filename}: {e}")
            return []
    
    def save(self, filename: str, records: List[NotebookRecord]) -> bool:
        """
        Save records to a notebook file.
        
        Args:
            filename: The name of the notebook file
            records: The list of NotebookRecord objects to save
            
        Returns:
            True if successful, False otherwise
        """
        file_path = self._get_file_path(filename)
        
        try:
            data = [record.to_dict() for record in records]
            
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            return True
        except Exception as e:
            print(f"Error saving notebook {filename}: {e}")
            return False
    
    def update(self, filename: str, new_record: NotebookRecord) -> bool:
        """
        Update a notebook with a new record.
        
        Args:
            filename: The name of the notebook file
            new_record: The new NotebookRecord to add
            
        Returns:
            True if successful, False otherwise
        """
        records = self.load(filename)
        
        # Add the new record
        records.append(new_record)
        
        # Save the updated records
        return self.save(filename, records)
    
    def update_record(self, filename: str, question: str, answer: Optional[str] = None, reason: Optional[str] = None) -> bool:
        """
        Update a notebook with a new question-answer pair.
        
        Args:
            filename: The name of the notebook file
            question: The question
            answer: The answer (optional)
            reason: The reasoning (optional)
            
        Returns:
            True if successful, False otherwise
        """
        new_record = NotebookRecord(question=question, answer=answer, reason=reason)
        return self.update(filename, new_record)
    
    def reset(self, filename: str) -> bool:
        """
        Reset a notebook (delete all records).
        
        Args:
            filename: The name of the notebook file
            
        Returns:
            True if successful, False otherwise
        """
        return self.save(filename, [])
    
    def list_notebooks(self) -> List[str]:
        """
        List all available notebooks.
        
        Returns:
            A list of notebook filenames (without extension)
        """
        if not os.path.exists(self.BASE_DIR):
            return []
        
        files = [f for f in os.listdir(self.BASE_DIR) if f.endswith('.json')]
        return [f[:-5] for f in files]  # Remove .json extension


# Create a singleton instance
notebook_manager = NotebookManager()


# Export functions for easy access

def load_notebook(filename: str) -> List[Dict[str, Any]]:
    """
    Load a notebook from a file.
    
    Args:
        filename: The name of the notebook file (without .json extension)
        
    Returns:
        A list of notebook records as dictionaries
    """
    records = notebook_manager.load(filename)
    return [record.to_dict() for record in records]


def update_notebook(filename: str, question: str, answer: Optional[str] = None, reason: Optional[str] = None) -> bool:
    """
    Update a notebook with a new question-answer pair.
    
    Args:
        filename: The name of the notebook file (without .json extension)
        question: The question
        answer: The answer (optional)
        reason: The reasoning (optional)
        
    Returns:
        True if successful, False otherwise
    """
    return notebook_manager.update_record(filename, question, answer, reason)


def save_notebook(filename: str, records: List[Dict[str, Any]]) -> bool:
    """
    Save records to a notebook file.
    
    Args:
        filename: The name of the notebook file (without .json extension)
        records: The list of notebook records as dictionaries
        
    Returns:
        True if successful, False otherwise
    """
    notebook_records = [NotebookRecord.from_dict(record) for record in records]
    return notebook_manager.save(filename, notebook_records)


def reset_notebook(filename: str) -> bool:
    """
    Reset a notebook (delete all records).
    
    Args:
        filename: The name of the notebook file (without .json extension)
        
    Returns:
        True if successful, False otherwise
    """
    return notebook_manager.reset(filename)


def list_notebooks() -> List[str]:
    """
    List all available notebooks.
    
    Returns:
        A list of notebook filenames (without extension)
    """
    return notebook_manager.list_notebooks() 