"""
Configuration package for the application.

This package provides configuration settings for the application.
"""

from config.settings import get_default_chat_model, ModelFactory, ModelProvider, WORKSPACE_DIR

__all__ = [
    'get_default_chat_model',
    'ModelFactory',
    'ModelProvider',
    'WORKSPACE_DIR'
] 