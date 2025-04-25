"""
Configuration settings for the application.

This module provides configuration settings for the application.
"""

import os
from enum import Enum
from typing import Dict, Any, Optional

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI


class ModelProvider(Enum):
    """Supported model providers."""
    OPENAI = "openai"
    DEEPSEEK = "deepseek"


class ModelFactory:
    """Factory for creating language model instances."""
    
    @staticmethod
    def create_chat_model(
        provider: Optional[ModelProvider] = None,
        model_name: Optional[str] = None,
        temperature: float = 0,
        **kwargs
    ) -> BaseChatModel:
        """
        Create a chat model based on the specified provider.
        
        Args:
            provider: The model provider (if None, determined from environment)
            model_name: Name of the model to use
            temperature: Temperature for generation
            **kwargs: Additional arguments to pass to the model
            
        Returns:
            A ChatModel instance
        """
        # Determine provider from environment if not specified
        if provider is None:
            if os.environ.get("DEEPSEEK_API_KEY"):
                provider = ModelProvider.DEEPSEEK
            else:
                provider = ModelProvider.OPENAI
        
        # Set default model names if not specified
        if model_name is None:
            if provider == ModelProvider.OPENAI:
                model_name = "gpt-4-turbo"
            elif provider == ModelProvider.DEEPSEEK:
                model_name = "deepseek-chat"
        
        # Create the appropriate model
        if provider == ModelProvider.OPENAI:
            return ChatOpenAI(
                model=model_name,
                temperature=temperature,
                **kwargs
            )
        elif provider == ModelProvider.DEEPSEEK:
            return ChatOpenAI(
                model=model_name,
                temperature=temperature,
                api_key=os.environ.get("DEEPSEEK_API_KEY"),
                base_url="https://api.deepseek.com/v1",
                **kwargs
            )
        
        raise ValueError(f"Unsupported model provider: {provider}")


def get_default_chat_model() -> BaseChatModel:
    """
    Get a chat model using environment settings.
    
    Returns:
        A ChatModel instance
    """
    return ModelFactory.create_chat_model()


# Application settings
WORKSPACE_DIR = os.environ.get("WORKSPACE_DIR", "workspace") 