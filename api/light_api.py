"""
Lightweight API router for basic functionality when heavy dependencies are not available.
This serves as a fallback when heavy_api.py cannot be loaded due to missing dependencies.
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/ping", tags=["Light System"])
async def ping_light():
    """Basic health check for the light API."""
    return {
        "status": "ok",
        "message": "Light API is operational (heavy dependencies not available)",
        "version": "1.0.0",
        "api_type": "light",
        "note": "Some advanced features may not be available"
    }

@router.get("/status", tags=["Light System"])
async def get_light_status():
    """Get basic system status."""
    return {
        "status": "operational",
        "api_type": "light",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "features": {
            "database": "not_configured",
            "data_processing": "unavailable",
            "ml_models": "unavailable",
            "heavy_compute": "unavailable"
        },
        "message": "Running in lightweight mode due to missing dependencies"
    }

@router.get("/astro/status", tags=["Light Astro"])
async def get_light_astro_status():
    """Basic astronomical data status (placeholder)."""
    return {
        "status": "not_available",
        "message": "Astronomical data processing requires heavy dependencies",
        "available_catalogs": [],
        "total_objects": 0
    }

@router.get("/api", tags=["Light Info"])
async def api_info():
    """Basic API information."""
    return {
        "message": "Scientific API (Light Mode)",
        "version": "1.0.0",
        "environment": "Lightweight fallback",
        "dependencies_status": "Some dependencies missing",
        "available_endpoints": [
            "/ping",
            "/status",
            "/astro/status",
            "/api"
        ]
    }
