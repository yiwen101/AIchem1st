import os
import json
from typing import Dict, Any, Optional


class Config:
    """Configuration manager for the tool planning agent."""
    
    DEFAULT_CONFIG = {
        "api_key": "",
        "model": "deepseek-chat",
        "temperature": 0.7,
        "max_tokens": 2000,
        "logging_level": "INFO",
        "api_base_url": "https://api.deepseek.com/v1/chat/completions",
    }
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to the configuration file. If None, default values are used.
        """
        self.config = self.DEFAULT_CONFIG.copy()
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                loaded_config = json.load(f)
                self.config.update(loaded_config)
        
        # Override with environment variables if set
        for key in self.config:
            env_var = f"TOOL_AGENT_{key.upper()}"
            if env_var in os.environ:
                self.config[key] = os.environ[env_var]
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            key: The configuration key
            default: Default value if key is not found
            
        Returns:
            The configuration value
        """
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value.
        
        Args:
            key: The configuration key
            value: The value to set
        """
        self.config[key] = value
    
    def save(self, config_path: str) -> None:
        """
        Save the configuration to a file.
        
        Args:
            config_path: Path to save the configuration file
        """
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=4)
            
    def __getitem__(self, key: str) -> Any:
        return self.get(key)
    
    def __setitem__(self, key: str, value: Any) -> None:
        self.set(key, value) 