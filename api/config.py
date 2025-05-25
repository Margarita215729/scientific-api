"""
Configuration module for the Scientific API application.
This module centralizes all configuration settings and provides
utility functions for managing environment variables.
"""

import os
import sys
import logging
from typing import Dict, Any, Optional

# Configure logger
logger = logging.getLogger(__name__)

# Environment detection
IS_PRODUCTION = os.environ.get("VERCEL") == "1"
IS_DEVELOPMENT = not IS_PRODUCTION

# Python version info
PYTHON_VERSION = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

# API Configuration
API_TITLE = "Scientific API"
API_DESCRIPTION = "API для работы с астрономическими данными и анализа крупномасштабной структуры Вселенной"
API_VERSION = "1.0.0"

# Directory paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UI_DIR = os.path.join(BASE_DIR, "ui")

# Load environment variables from .env if present and not in production
if not IS_PRODUCTION:
    try:
        from dotenv import load_dotenv
        load_dotenv()
        logger.info("Loaded environment variables from .env file")
    except ImportError:
        logger.warning("python-dotenv not installed, skipping .env file loading")
    except Exception as e:
        logger.warning(f"Error loading .env file: {e}")

def get_env_var(name: str, default: Any = None, required: bool = False) -> Optional[str]:
    """
    Get an environment variable with optional default value and validation.
    """
    value = os.environ.get(name, default)
    
    if required and value is None:
        logger.warning(f"Required environment variable {name} is not set")
    
    if value is not None:
        # Log the name of the env var but not its value for security
        logger.info(f"Environment variable {name} is set")
    else:
        logger.info(f"Environment variable {name} is not set, using default")
    
    return value

# Required API keys and tokens
ADSABS_TOKEN = get_env_var("ADSABS_TOKEN")
GOOGLE_CLIENT_ID = get_env_var("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = get_env_var("GOOGLE_CLIENT_SECRET")
GOOGLE_REFRESH_TOKEN = get_env_var("GOOGLE_REFRESH_TOKEN")
SERPAPI_KEY = get_env_var("SERPAPI_KEY")

# Application settings
DEBUG = get_env_var("DEBUG", "true" if IS_DEVELOPMENT else "false").lower() == "true"
LOG_LEVEL = get_env_var("LOG_LEVEL", "DEBUG" if DEBUG else "INFO")

def get_config() -> Dict[str, Any]:
    """
    Get all configuration settings as a dictionary.
    Sensitive values are redacted.
    """
    return {
        "environment": "production" if IS_PRODUCTION else "development",
        "python_version": PYTHON_VERSION,
        "api": {
            "title": API_TITLE,
            "version": API_VERSION,
            "debug": DEBUG,
        },
        "paths": {
            "base_dir": BASE_DIR,
            "ui_dir": UI_DIR,
        },
        "tokens": {
            "adsabs_token": "***redacted***" if ADSABS_TOKEN else None,
            "google_client_id": "***redacted***" if GOOGLE_CLIENT_ID else None,
            "google_refresh_token": "***redacted***" if GOOGLE_REFRESH_TOKEN else None,
            "serpapi_key": "***redacted***" if SERPAPI_KEY else None,
        },
    } 