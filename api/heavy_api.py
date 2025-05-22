"""
Heavy compute service API endpoints.
This service handles data processing, machine learning, and other resource-intensive operations.
It should be deployed to a platform without strict size limitations (e.g., Google Cloud Run, AWS Lambda Container).
"""

from fastapi import FastAPI, APIRouter, HTTPException
from fastapi.responses import JSONResponse
import os
from typing import Dict, Any, List, Optional

# Create FastAPI app instance
app = FastAPI(
    title="Scientific API Heavy Compute Service",
    description="Heavy compute service for data processing and machine learning",
    version="1.0.0"
)

router = APIRouter()

@router.get("/ping")
async def ping():
    """Health check endpoint."""
    return {
        "status": "ok",
        "message": "Heavy compute service is up and running",
        "service": "heavy-compute"
    }

@router.get("/datasets/search")
async def search_datasets(
    query: str,
    max_results: int = 10
):
    """Search for datasets using various sources."""
    try:
        # Import heavy dependencies only when needed
        from utils.dataset_fetcher import (
            fetch_from_arxiv,
            fetch_from_openml,
            fetch_from_biorxiv
        )
        
        results = []
        # Add results from different sources
        results.extend(await fetch_from_arxiv(query, max_results))
        results.extend(await fetch_from_openml(query, max_results))
        results.extend(await fetch_from_biorxiv(query, max_results))
        
        return {
            "status": "ok",
            "count": len(results),
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/ads/search")
async def search_ads(
    query: str,
    max_results: int = 10
):
    """Search ADS for publications."""
    try:
        from utils.ads_astronomy import search_by_object
        
        results = await search_by_object(query, max_results)
        return {
            "status": "ok",
            "count": len(results),
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/ml/predict")
async def ml_predict(
    data: List[float],
    model_type: str = "default"
):
    """Make predictions using machine learning models."""
    try:
        from utils.ml_models import get_prediction
        
        result = await get_prediction(data, model_type)
        return {
            "status": "ok",
            "prediction": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/astro/analyze")
async def analyze_astro_data(
    data: List[Dict[str, Any]],
    analysis_type: str
):
    """Analyze astronomical data."""
    try:
        from utils.astronomy_catalogs import analyze_data
        
        result = await analyze_data(data, analysis_type)
        return {
            "status": "ok",
            "analysis": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Include the router in the app
app.include_router(router) 