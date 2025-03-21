"""
Query LLM tool implementation for interfacing with language models.
"""

from typing import Dict, Any, Optional

from src.models.interfaces import Tool
from src.adapter.llm_adapter import LLMAdapter
from src.utils.config import Config
from src.utils.logging import LoggingManager

logger = LoggingManager.get_logger()


def create_query_function(llm_adapter: Optional[LLMAdapter] = None):
    """
    Create a function for querying an LLM.
    
    Args:
        llm_adapter: An existing LLM adapter to use, or None to create a new one
        
    Returns:
        Function for querying the LLM
    """
    # Create an LLM adapter if one wasn't provided
    adapter = llm_adapter
    
    def query_llm(
        query: str, 
        system_prompt: str = "You are a helpful AI assistant that thinks deeply about questions and provides accurate, detailed responses.",
        temperature: float = 0.7,
        deep_thinking: bool = True
    ) -> Dict[str, Any]:
        """
        Query the language model with a prompt and get a response.
        
        Args:
            query: The query to send to the model
            system_prompt: Instructions for the model
            temperature: Temperature setting for response generation
            deep_thinking: Whether to enable deep thinking mode
            max_tokens: Maximum number of tokens in the response (uses default if None)
            
        Returns:
            Dictionary containing the response from the LLM
        """
        logger.info(f"Querying LLM: {query}")
        
        try:
            # Format the prompt with system instructions and query
            if system_prompt:
                full_prompt = f"{system_prompt}\n\nQuery: {query}"
            else:
                full_prompt = query
                
            # Directly use the adapter's generate method with the provided parameters
            response = adapter.generate(
                prompt=full_prompt,
                temperature=temperature,
                deep_thinking=deep_thinking
            )
            
            # Estimate token count (rough approximation)
            word_count = len(response.split())
            token_estimate = int(word_count * 1.3)  # Rough token estimate
            
            return {
                "response": response
            }
        except Exception as e:
            logger.error(f"Error querying LLM: {str(e)}")
            return {
                "error": str(e),
                "success": False
            }
    
    return query_llm


def create_query_llm_tool(llm_adapter: Optional[LLMAdapter] = None) -> Tool:
    """
    Create a query LLM tool instance.
    
    Args:
        llm_adapter: An existing LLM adapter to use, or None to create a new one
        
    Returns:
        Tool instance for querying an LLM
    """
    return Tool(
        name="query_llm",
        description="Query a thinking language model with a prompt in a new thread and get a response. You will need to provide the full context of the query in the prompt.",
        parameters={
            "query": {
                "type": "string",
                "description": "The query or prompt to send to the model"
            },
            "system_prompt": {
                "type": "string",
                "description": "Instructions for the model (optional)",
                "default": "You are a helpful AI assistant that thinks deeply about questions and provides accurate, detailed responses."
            },
            "temperature": {
                "type": "number",
                "description": "Temperature setting (0.0-1.0) for response generation",
                "default": 0.7
            },
            "deep_thinking": {
                "type": "boolean",
                "description": "Whether to enable deep thinking mode for more thorough responses",
                "default": True
            }
        },
        function=create_query_function(llm_adapter)
    ) 