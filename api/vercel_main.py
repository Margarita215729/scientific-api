"""
Lightweight entry point for Vercel deployment
"""

import os
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Create minimal FastAPI app
app = FastAPI(
    title="Scientific Data Platform",
    description="Comprehensive platform for scientific data management",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Welcome page"""
    return {
        "message": "Welcome to Scientific Data Platform",
        "status": "online",
        "version": "1.0.0",
        "features": [
            "Data Collection from arXiv, SerpAPI, ADS, NASA, SDSS",
            "Advanced Data Cleaning & Transformation",
            "API Integrations Testing",
            "Interactive Dashboard"
        ],
        "endpoints": {
            "ping": "/ping",
            "api_info": "/api",
            "status": "/api/status",
            "docs": "/docs"
        }
    }

@app.get("/ping")
async def ping():
    """Health check endpoint"""
    return {
        "status": "ok",
        "message": "Scientific Data Platform is running",
        "version": "1.0.0",
        "environment": "production"
    }

@app.get("/api")
async def api_info():
    """API information"""
    return {
        "message": "Scientific Data Platform API",
        "version": "1.0.0",
        "environment": "production",
        "endpoints": {
            "dashboard": "/",
            "ping": "/ping",
            "docs": "/docs",
            "integrations": "/api/test/integrations",
            "templates": "/api/test/templates"
        }
    }

# Simple API endpoints for Vercel (no heavy imports)
@app.get("/api/test/simple")
async def test_simple():
    """Simple test endpoint"""
    return {
        "status": "ok",
        "platform": "Vercel",
        "timestamp": "2025-08-30",
        "integrations": {
            "arxiv": "Available",
            "serpapi": "Available" if os.getenv("SERPAPI_KEY") else "Not configured",
            "ads": "Available" if os.getenv("ADSABS_TOKEN") else "Not configured"
        }
    }

# Basic data endpoints without heavy dependencies
@app.get("/api/status")
async def get_status():
    """Get platform status"""
    return {
        "platform": "Scientific Data Platform",
        "status": "operational",
        "features": [
            "Data Collection (arXiv, SerpAPI, ADS, NASA, SDSS)",
            "Data Cleaning & Transformation",
            "API Integrations Testing",
            "Interactive Dashboard"
        ],
        "integrations": {
            "arxiv": "Available",
            "serpapi": "Available" if os.getenv("SERPAPI_KEY") else "Not configured",
            "ads": "Available" if os.getenv("ADSABS_TOKEN") else "Not configured",
            "nasa": "Available",
            "sdss": "Available"
        }
    }

# This is the entry point for Vercel
handler = app
