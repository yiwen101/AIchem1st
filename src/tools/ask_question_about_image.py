import os
from typing import Dict, Any
from tools.base_tool import BaseTool

class AskQuestionAboutImage(BaseTool):
    """Tool to ask questions about an image and get human input answers."""
    
    @property
    def description(self) -> str:
        return "Asks a specific `what` type question about an image and gets a human response via terminal input. The human may refuse to answer if the question is not primitive."
    
    @property
    def input_schema(self) -> Dict[str, Dict[str, Any]]:
        return {
            "image_path": {
                "type": "string",
                "description": "Path to the image file to ask questions about",
                "required": True
            },
            "question": {
                "type": "string",
                "description": "The specific question to ask about the image",
                "required": True
            }
        }
    
    @property
    def output_schema(self) -> Dict[str, Any]:
        return {
            "answer": {
                "type": "string",
                "description": "The human-provided answer to the question about the image"
            },
            "image_exists": {
                "type": "boolean",
                "description": "Whether the specified image file exists"
            }
        }
    
    def execute(self, image_path: str, question: str) -> Dict[str, Any]:
        """
        Ask a question about an image and get a human input answer.
        
        Args:
            image_path (str): Path to the image file
            question (str): The question to ask about the image
            
        Returns:
            dict: The human-provided answer and image existence status
        """
        # Check if the image exists
        image_exists = os.path.exists(image_path)
        
        # Print information to the terminal
        print("\n" + "="*50)
        print(f"HUMAN EVALUATION REQUIRED")
        print("="*50)
        print(f"Question about image: {question}")
        print(f"Image path: {image_path}")
        
        if not image_exists:
            print(f"WARNING: Image file does not exist at the specified path!")
        
        # Get human input
        print("-"*50)
        answer = input("Please provide your answer to this question: ")
        print("="*50 + "\n")
        
        return {
            "answer": answer,
            "image_exists": image_exists
        } 