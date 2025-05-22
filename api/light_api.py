"""
Lightweight API endpoints that don't require heavy dependencies.
These endpoints will be deployed to Vercel.
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
import os
from typing import Dict, Any

router = APIRouter()

@router.get("/ping")
async def ping():
    """Basic health check endpoint."""
    return {"status": "ok", "message": "Lightweight API is up and running"}

@router.get("/status")
async def status():
    """Get API status and configuration."""
    return {
        "status": "ok",
        "version": "1.0.0",
        "environment": os.getenv("ENVIRONMENT", "development"),
        "heavy_compute_url": os.getenv("HEAVY_COMPUTE_URL", "http://localhost:8000")
    }

@router.get("/astro/status")
async def astro_status():
    """Get status of astronomical data services."""
    return {
        "status": "ok",
        "services": {
            "basic": "available",
            "heavy_compute": "requires separate service"
        }
    }

@router.get("/ads/basic")
async def ads_basic():
    """Basic ADS service status."""
    return {
        "status": "ok",
        "message": "Basic ADS service is available",
        "note": "For full ADS functionality, use the heavy compute service"
    } 