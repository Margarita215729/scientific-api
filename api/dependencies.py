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

def get_sample_galaxies() -> List[Dict[str, Any]]:
    """
    Return sample galaxy data that doesn't require pandas.
    This function provides a fallback when pandas is not available.
    """
    return [
        {"RA": 150.1, "DEC": 2.2, "redshift": 0.5, "source": "SDSS", "X": 100.0, "Y": 200.0, "Z": 300.0},
        {"RA": 151.2, "DEC": 2.3, "redshift": 0.6, "source": "SDSS", "X": 110.0, "Y": 210.0, "Z": 310.0},
        {"RA": 149.3, "DEC": 2.1, "redshift": 0.4, "source": "Euclid", "X": 90.0, "Y": 190.0, "Z": 290.0},
        {"RA": 152.4, "DEC": 2.4, "redshift": 0.7, "source": "DESI", "X": 120.0, "Y": 220.0, "Z": 320.0},
        {"RA": 153.5, "DEC": 2.5, "redshift": 0.8, "source": "DESI", "X": 130.0, "Y": 230.0, "Z": 330.0},
        {"RA": 148.6, "DEC": 2.0, "redshift": 0.3, "source": "DES", "X": 80.0, "Y": 180.0, "Z": 280.0},
        {"RA": 147.7, "DEC": 1.9, "redshift": 0.2, "source": "DES", "X": 70.0, "Y": 170.0, "Z": 270.0},
        {"RA": 146.8, "DEC": 1.8, "redshift": 0.1, "source": "DES", "X": 60.0, "Y": 160.0, "Z": 260.0},
        {"RA": 154.9, "DEC": 2.6, "redshift": 0.9, "source": "Euclid", "X": 140.0, "Y": 240.0, "Z": 340.0},
        {"RA": 156.0, "DEC": 2.7, "redshift": 1.0, "source": "Euclid", "X": 150.0, "Y": 250.0, "Z": 350.0},
    ] 