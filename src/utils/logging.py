import logging
import sys
import os
from datetime import datetime
from typing import Optional


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """
    Set up logging for the tool planning agent.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to the log file. If None, a timestamped log file will be created.
        
    Returns:
        A configured logger instance
    """
    # Convert level string to logging level
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")
    
    # Create logger
    logger = logging.getLogger("tool_planning_agent")
    logger.setLevel(numeric_level)
    
    # Clear existing handlers
    if logger.handlers:
        logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Generate default log filename with timestamp if not provided
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"logs/log_{timestamp}.log"
    
    # Create logs directory if it doesn't exist
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


# Global logger instance
logger = setup_logging()


class LoggingManager:
    """Manager for logging functionality."""
    
    @staticmethod
    def configure(level: str = "INFO", log_file: Optional[str] = None) -> None:
        """
        Configure the global logger.
        
        Args:
            level: Logging level
            log_file: Path to log file. If None, a timestamped log file will be created.
        """
        global logger
        logger = setup_logging(level, log_file)
    
    @staticmethod
    def get_logger() -> logging.Logger:
        """
        Get the global logger instance.
        
        Returns:
            The global logger
        """
        return logger 