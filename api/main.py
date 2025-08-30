"""
Main entry point for Vercel deployment
"""

import os
import sys

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set environment variables for Vercel
os.environ.setdefault('DB_TYPE', 'mongodb')
os.environ.setdefault('DEBUG', 'false')
os.environ.setdefault('LOG_LEVEL', 'INFO')

# Import the FastAPI app
from api.index import app

# This is the entry point for Vercel
handler = app
