"""
Static files handling module for the FastAPI application.
"""

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import pathlib
import os
import logging

# Configure logger
logger = logging.getLogger(__name__)

def setup_static_files(app: FastAPI):
    """
    Set up static files based on the environment.
    For Vercel, we use a special configuration to serve UI files.
    """
    try:
        # Get the directory of the current file
        current_dir = pathlib.Path(__file__).parent.parent
        ui_dir = current_dir / "ui"
        
        logger.info(f"Current directory: {current_dir}")
        logger.info(f"UI directory: {ui_dir}")
        
        # Check if UI directory exists
        if not ui_dir.exists():
            logger.warning(f"UI directory not found at {ui_dir}. Checking alternative locations...")
            
            # Try an alternative location (Vercel's build output directory)
            if os.environ.get("VERCEL") == "1":
                # On Vercel, the build output might be in a different location
                vercel_dir = pathlib.Path(os.environ.get("VERCEL_OUTPUT_DIR", "/var/task"))
                ui_dir = vercel_dir / "ui"
                logger.info(f"Trying Vercel UI directory: {ui_dir}")
        
        # Check if we're running on Vercel
        if os.environ.get("VERCEL") == "1":
            # On Vercel, static files are served via routes configuration
            # But we still need to mount them for FastAPI's reference
            logger.info("Running on Vercel - static files will be served via Vercel's routing")
            
            # List files in the UI directory for debugging
            try:
                if ui_dir.exists():
                    logger.info(f"Contents of {ui_dir}:")
                    for item in ui_dir.iterdir():
                        logger.info(f"  - {item.name} {'(dir)' if item.is_dir() else '(file)'}")
                else:
                    logger.warning(f"UI directory does not exist: {ui_dir}")
            except Exception as e:
                logger.error(f"Error listing UI directory: {e}")
        else:
            # Local development - mount static files directly
            logger.info(f"Setting up static files from {ui_dir}")
            try:
                if ui_dir.exists():
                    app.mount("/static", StaticFiles(directory=str(ui_dir)), name="static")
                    logger.info("Static files mounted successfully")
                else:
                    logger.warning(f"Cannot mount static files - directory does not exist: {ui_dir}")
            except Exception as e:
                logger.error(f"Error mounting static files: {e}")
        
        return ui_dir
    
    except Exception as e:
        logger.error(f"Error in setup_static_files: {e}")
        # Fallback to a sensible default
        current_dir = pathlib.Path(__file__).parent.parent
        ui_dir = current_dir / "ui"
        logger.info(f"Using fallback UI directory: {ui_dir}")
        return ui_dir 