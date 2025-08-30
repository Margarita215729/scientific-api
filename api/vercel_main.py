"""
Lightweight entry point for Vercel deployment
"""

import os
import sys
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

# Set environment variables for Vercel
os.environ.setdefault('DB_TYPE', 'mongodb')
os.environ.setdefault('DEBUG', 'false')
os.environ.setdefault('LOG_LEVEL', 'INFO')

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

# Setup templates
ui_dir = Path(__file__).parent.parent / "ui"
templates = Jinja2Templates(directory=str(ui_dir))

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serve the main dashboard"""
    try:
        return templates.TemplateResponse("dashboard.html", {"request": request})
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": "Error rendering dashboard", "error": str(e)}
        )

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

# Include lightweight routers only for Vercel
try:
    from api.test_integrations import router as test_router
    app.include_router(test_router)
except ImportError:
    pass

try:
    from api.light_api import router as light_router
    app.include_router(light_router)
except ImportError:
    pass

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
