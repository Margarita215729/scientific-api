"""
Dependencies management for the FastAPI application.
This module helps handle optional dependencies and provides fallback implementations.
"""

import importlib
import logging
from typing import Optional, Dict, Any, List

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_dependency(module_name: str) -> bool:
    """Check if a Python module is available."""
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False

# Dictionary of required dependencies and their fallback status
DEPENDENCIES = {
    "pandas": {
        "available": check_dependency("pandas"),
        "required_for": ["data processing", "catalog operations"]
    },
    "numpy": {
        "available": check_dependency("numpy"),
        "required_for": ["numerical calculations", "coordinate transformations"]
    },
    "requests": {
        "available": check_dependency("requests"),
        "required_for": ["ADS API", "external data sources"]
    },
    "scikit-learn": {
        "available": check_dependency("sklearn"),
        "required_for": ["machine learning models"]
    },
    "matplotlib": {
        "available": check_dependency("matplotlib"),
        "required_for": ["data visualization"]
    }
}

def get_dependency_status() -> Dict[str, Any]:
    """Get the status of all dependencies."""
    return DEPENDENCIES

def print_dependency_status():
    """Print the status of all dependencies to the console."""
    logger.info("Checking dependencies...")
    
    for name, info in DEPENDENCIES.items():
        status = "✅ Available" if info["available"] else "❌ Missing"
        logger.info(f"{name}: {status} (needed for: {', '.join(info['required_for'])})")
    
    logger.info("Dependency check complete.")