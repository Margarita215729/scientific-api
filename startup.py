#!/usr/bin/env python3
"""
Startup script for Azure Container Instance.
This script runs data preprocessing before starting the main API server.
"""

import asyncio
import os
import sys
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def preprocess_data():
    """Run data preprocessing."""
    try:
        logger.info("="*50)
        logger.info("STARTING DATA PREPROCESSING")
        logger.info("="*50)
        
        from utils.data_preprocessor import AstronomicalDataPreprocessor
        
        preprocessor = AstronomicalDataPreprocessor()
        results = await preprocessor.preprocess_all_catalogs()
        
        logger.info("="*50)
        logger.info("DATA PREPROCESSING COMPLETED")
        logger.info(f"Total objects: {results['total_objects']}")
        
        for catalog, info in results["catalogs"].items():
            status = info.get('status', 'unknown')
            objects = info.get('objects', 0)
            logger.info(f"  {catalog}: {status} ({objects} objects)")
        
        logger.info("="*50)
        
        return True
        
    except Exception as e:
        logger.error(f"Data preprocessing failed: {e}")
        logger.error("Continuing with API startup...")
        return False

def start_api_server():
    """Start the FastAPI server."""
    logger.info("Starting FastAPI server...")
    
    try:
        import uvicorn
        from api.heavy_api import app
        
        # Get host and port from environment
        host = os.getenv("HOST", "0.0.0.0")
        port = int(os.getenv("PORT", "8000"))
        workers = int(os.getenv("WORKERS", "1"))
        
        logger.info(f"Starting server on {host}:{port} with {workers} workers")
        
        uvicorn.run(
            app,
            host=host,
            port=port,
            workers=workers,
            log_level="info"
        )
        
    except Exception as e:
        logger.error(f"Failed to start API server: {e}")
        sys.exit(1)

async def main():
    """Main startup function."""
    logger.info("Azure Container Instance startup initiated")
    logger.info(f"Environment: {os.getenv('ENVIRONMENT', 'production')}")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Working directory: {os.getcwd()}")
    
    # Check if we should skip preprocessing (for development)
    skip_preprocessing = os.getenv("SKIP_PREPROCESSING", "false").lower() == "true"
    
    if skip_preprocessing:
        logger.info("Skipping data preprocessing (SKIP_PREPROCESSING=true)")
    else:
        # Run data preprocessing
        preprocessing_success = await preprocess_data()
        
        if preprocessing_success:
            logger.info("Data preprocessing completed successfully")
        else:
            logger.warning("Data preprocessing failed, but continuing with API startup")
    
    # Start the API server
    start_api_server()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutdown initiated by user")
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        sys.exit(1) 