"""
Logger module for AIchem1st application.

This module provides logging functionality that logs to log/timestamp.log.
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

# Create log directory if it doesn't exist
os.makedirs("log", exist_ok=True)


# Set up the logger
class Logger:
    """Logger class for AIchem1st application."""

    _instance = None

    def __new__(cls):
        """Create a singleton instance."""
        if cls._instance is None:
            cls._instance = super(Logger, cls).__new__(cls)
            cls._instance.setup_logger()
        return cls._instance

    def setup_logger(self):
        """Set up the logger with the current timestamp."""
        # Create a logger
        self.logger = logging.getLogger("aichem1st")

        # Create a timestamped file handler
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = Path("logs")
        if not log_dir.exists():
            log_dir.mkdir(parents=True)
        log_file = log_dir / f"{timestamp}.log"
        file_handler = logging.FileHandler(log_file)

        # Create a console handler
        console_handler = logging.StreamHandler()

        # Create a formatter and add it to the handlers
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add the handlers to the logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        self.log_info(f"Logger initialized. Logging to {log_file}")

    def log_info(self, message: str):
        """Log an info message."""
        self.logger.info(message)

    def log_warning(self, message: str):
        """Log a warning message."""
        self.logger.warning(message)

    def log_error(self, message: str):
        """Log an error message."""
        self.logger.error(message)

    def log_debug(self, message: str):
        """Log a debug message."""
        self.logger.debug(message)

    def log_llm_prompt(self, prompt: str):
        """Log an LLM prompt."""
        self.logger.info(f"LLM Prompt: {prompt}")

    def log_llm_response(self, response: any):
        """Log an LLM response."""
        self.logger.info(f"LLM Response: {response}")

    def log_tool_call(self, tool_name: str, parameters: dict):
        """Log a tool call."""
        self.logger.info(f"Tool Call: {tool_name} with parameters: {parameters}")

    def log_tool_result(self, tool_name: str, result: any):
        """Log a tool result."""
        self.logger.info(f"Tool Result: {tool_name} returned: {result}")

    def log_exception(self, exc: Exception, context: Optional[str] = None):
        """Log an exception."""
        if context:
            self.logger.error(f"Exception in {context}: {str(exc)}", exc_info=True)
        else:
            self.logger.error(f"Exception: {str(exc)}", exc_info=True)


# Create a singleton instance
logger = Logger()
