"""
Lightweight API endpoints that don't require heavy dependencies.
These endpoints will be deployed to Vercel.
"""

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import JSONResponse
import os
from typing import Dict, Any
import httpx

router = APIRouter()

HEAVY_COMPUTE_URL = os.getenv("HEAVY_COMPUTE_URL", "http://localhost:8001")

# Universal proxy route for heavy endpoints
def is_heavy_path(path: str) -> bool:
    heavy_paths = [
        "astro/galaxies",
        "astro/statistics",
        "ads",
        "datasets/search",
        "ml/predict",
        "astro/analyze"
    ]
    return path in heavy_paths

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
        "heavy_compute_url": HEAVY_COMPUTE_URL
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

@router.api_route("/{proxy_path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy_to_heavy(request: Request, proxy_path: str):
    if not is_heavy_path(proxy_path):
        return JSONResponse({"detail": "Not Found"}, status_code=404)

    url = f"{HEAVY_COMPUTE_URL}/{proxy_path}"
    method = request.method
    headers = dict(request.headers)
    body = await request.body()

    async with httpx.AsyncClient() as client:
        resp = await client.request(
            method, url, headers=headers, content=body, params=dict(request.query_params)
        )
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            headers=resp.headers
        ) 
