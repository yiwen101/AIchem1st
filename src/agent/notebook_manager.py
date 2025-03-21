import json
import os
import uuid
from datetime import datetime

class NotebookManager:
    def __init__(self, notebook_path=None):
        """
        Initialize the NotebookManager to track question-answer history.
        
        Args:
            notebook_path (str, optional): Path to the notebook JSON file
        """
        from ..utils.config import Config
        self.notebook_path = notebook_path or Config.NOTEBOOK_PATH
        self.notebook = self._load_notebook()
    
    def _load_notebook(self):
        """Load the notebook from the JSON file or create a new one."""
        if os.path.exists(self.notebook_path):
            try:
                with open(self.notebook_path, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print(f"Error loading notebook file. Creating a new one.")
                return {"entries": []}
        else:
            return {"entries": []}
    
    def _save_notebook(self):
        """Save the notebook to the JSON file."""
        with open(self.notebook_path, 'w') as f:
            json.dump(self.notebook, f, indent=2)
    
    def add_entry(self, question, answer, tools_used=None, parent_id=None):
        """
        Add a new entry to the notebook.
        
        Args:
            question (str): The question asked
            answer (str): The answer provided
            tools_used (list, optional): List of tools used to answer
            parent_id (str, optional): ID of the parent question if this is a sub-question
            
        Returns:
            str: ID of the created entry
        """
        entry_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        entry = {
            "id": entry_id,
            "question": question,
            "answer": answer,
            "timestamp": timestamp,
            "tools_used": tools_used or []
        }
        
        if parent_id:
            # This is a sub-question, add it to the parent entry
            parent_entry = self.get_entry_by_id(parent_id)
            if parent_entry:
                if "sub_questions" not in parent_entry:
                    parent_entry["sub_questions"] = []
                parent_entry["sub_questions"].append(entry)
            else:
                # Parent not found, add as a top-level entry
                self.notebook["entries"].append(entry)
        else:
            # This is a top-level question
            self.notebook["entries"].append(entry)
        
        self._save_notebook()
        return entry_id
    
    def get_entry_by_id(self, entry_id):
        """
        Retrieve an entry by its ID.
        
        Args:
            entry_id (str): The ID of the entry to retrieve
            
        Returns:
            dict: The entry if found, None otherwise
        """
        # Search in top-level entries
        for entry in self.notebook["entries"]:
            if entry["id"] == entry_id:
                return entry
            
            # Also search in sub-questions
            if "sub_questions" in entry:
                for sub_entry in entry["sub_questions"]:
                    if sub_entry["id"] == entry_id:
                        return sub_entry
        
        return None
    
    def get_all_entries(self):
        """Get all entries in the notebook."""
        return self.notebook["entries"]
    
    def search_entries(self, query, limit=5):
        """
        Search for entries containing the query in questions or answers.
        
        Args:
            query (str): The search query
            limit (int, optional): Maximum number of results
            
        Returns:
            list: Matching entries
        """
        results = []
        query = query.lower()
        
        def search_in_entries(entries):
            for entry in entries:
                if (query in entry["question"].lower() or 
                    query in entry["answer"].lower()):
                    results.append(entry)
                
                # Also search in sub-questions
                if "sub_questions" in entry:
                    search_in_entries(entry["sub_questions"])
        
        search_in_entries(self.notebook["entries"])
        return results[:limit]
    
    def get_recent_entries(self, limit=5):
        """
        Get the most recent entries.
        
        Args:
            limit (int, optional): Maximum number of entries to return
            
        Returns:
            list: Recent entries
        """
        flattened_entries = []
        
        # Flatten the notebook structure for sorting
        for entry in self.notebook["entries"]:
            flattened_entries.append(entry)
            if "sub_questions" in entry:
                for sub_entry in entry["sub_questions"]:
                    flattened_entries.append(sub_entry)
        
        # Sort by timestamp (most recent first)
        sorted_entries = sorted(
            flattened_entries, 
            key=lambda x: x.get("timestamp", ""), 
            reverse=True
        )
        
        return sorted_entries[:limit] 