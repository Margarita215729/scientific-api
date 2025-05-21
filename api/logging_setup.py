"""
Logging setup module for the FastAPI application.
"""

import logging
import sys
import os
from typing import Optional, Dict, Any

def setup_logging(
    level: str = "INFO",
    log_format: Optional[str] = None,
    app_name: str = "scientific-api"
) -> logging.Logger:
    """
    Set up logging for the application.
    
    Args:
        level: The logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: The log format. If None, a default format will be used.
        app_name: The name of the application logger
        
    Returns:
        The configured logger
    """
    # Convert string level to logging level
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Default format includes timestamp, level, name and message
    if log_format is None:
        if os.environ.get("VERCEL") == "1":
            # Simplified format for Vercel (they add their own timestamps)
            log_format = "%(levelname)s - %(name)s - %(message)s"
        else:
            # More detailed format for local development
            log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Configure the root logger
    logging.basicConfig(
        level=numeric_level,
        format=log_format,
        stream=sys.stdout  # Vercel captures stdout
    )
    
    # Create and configure our app logger
    logger = logging.getLogger(app_name)
    logger.setLevel(numeric_level)
    
    # Prevent propagation to avoid duplicate logs
    logger.propagate = False
    
    # Create a handler if logger doesn't have one
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(handler)
    
    return logger

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the given name.
    
    Args:
        name: The name of the logger, typically the module name
        
    Returns:
        A configured logger
    """
    return logging.getLogger(f"scientific-api.{name}")

def log_environment_info(logger: logging.Logger) -> Dict[str, Any]:
    """
    Log information about the environment.
    
    Args:
        logger: The logger to use
        
    Returns:
        A dictionary with environment information
    """
    # Get Python version
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    
    # Determine environment
    is_vercel = os.environ.get("VERCEL") == "1"
    env_name = "Production (Vercel)" if is_vercel else "Development"
    
    # Log the info
    logger.info(f"Python version: {py_version}")
    logger.info(f"Environment: {env_name}")
    
    # Return as dict for possible use in API responses
    return {
        "python_version": py_version,
        "environment": env_name,
        "is_production": is_vercel
    } 