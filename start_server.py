#!/usr/bin/env python3
"""
Simple script to start the Scientific API server
"""

# Load environment variables first
import os
import sys

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
import load_env

# Now import and run the app
import uvicorn

if __name__ == "__main__":
    print("Starting Scientific API server...")
    # Import app string to ensure env vars are loaded before app initialization
    uvicorn.run(
        "api.index:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Disable reload to prevent env var issues
        log_level="info"
    )
