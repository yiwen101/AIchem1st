import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    # API Keys
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
    
    # Agent settings
    MAX_RECURSION_DEPTH = 20
    DEFAULT_TIMEOUT = 60  # seconds
    
    # Tool settings
    TOOL_TIMEOUT = 30  # seconds
    
    # Notebook settings
    NOTEBOOK_PATH = os.getenv("NOTEBOOK_PATH", "notebook.json")
    
    # Model settings
    MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-chat") 