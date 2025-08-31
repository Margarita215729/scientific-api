#!/usr/bin/env python3
"""
Development server startup script.
Starts the FastAPI server with all necessary configuration.
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Start the development server."""
    print("🚀 Starting Scientific API Development Server...")
    
    # Set environment variables
    os.environ["PYTHONPATH"] = str(Path.cwd())
    os.environ["ENVIRONMENT"] = "development"
    
    # Start server
    try:
        cmd = [
            sys.executable, "-m", "uvicorn", 
            "api.index:app", 
            "--reload", 
            "--host", "0.0.0.0",
            "--port", "8000",
            "--log-level", "info"
        ]
        
        print(f"Starting server with command: {' '.join(cmd)}")
        print("Server will be available at: http://localhost:8000")
        print("API documentation: http://localhost:8000/docs")
        print("Press Ctrl+C to stop the server")
        
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
