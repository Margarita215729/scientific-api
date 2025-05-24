"""
Heavy compute service API endpoints for real astronomical data processing.
This service handles data processing, machine learning, and resource-intensive operations.
Designed for Microsoft Container Instances with 12 CPU cores and 20GB RAM.
"""

from fastapi import FastAPI, APIRouter, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse, FileResponse
import os
import logging
from typing import Dict, Any, List, Optional
import asyncio
from datetime import datetime
import json
import httpx

# Try to import heavy compute libraries
try:
    import pandas as pd
    import numpy as np
    HEAVY_LIBS_AVAILABLE = True
except ImportError:
    pd = None
    np = None
    HEAVY_LIBS_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get Azure API URL from environment
HEAVY_COMPUTE_URL = os.getenv("HEAVY_COMPUTE_URL", "")
USE_AZURE_API = bool(HEAVY_COMPUTE_URL) and not HEAVY_LIBS_AVAILABLE

logger.info(f"Heavy API initialized - HEAVY_LIBS_AVAILABLE: {HEAVY_LIBS_AVAILABLE}, USE_AZURE_API: {USE_AZURE_API}")

# Create FastAPI app instance
app = FastAPI(
    title="Scientific API Heavy Compute Service",
    description="Heavy compute service for real astronomical data processing and ML-ready dataset preparation",
    version="2.0.0"
)

router = APIRouter()

# Global storage for background tasks
background_tasks_status = {}

@router.get("/ping")
async def ping():
    """Health check endpoint."""
    return {
        "status": "ok",
        "message": "Heavy compute service is up and running",
        "service": "heavy-compute",
        "version": "2.0.0",
        "resources": {
            "cpu_cores": 12,
            "memory_gb": 20
        }
    }

@router.post("/astro/download")
async def download_astronomical_catalogs(background_tasks: BackgroundTasks):
    """
    Start downloading and processing real astronomical catalogs:
    - SDSS DR17 spectroscopic catalog
    - Euclid Q1 MER Final catalog  
    - DESI DR1 ELG clustering catalog
    - DES Y6 Gold catalog
    """
    import uuid
    task_id = str(uuid.uuid4())
    
    background_tasks.add_task(process_astronomical_data, task_id)
    background_tasks_status[task_id] = {
        "status": "started",
        "progress": 0,
        "message": "Initializing astronomical data download",
        "started_at": datetime.now().isoformat()
    }
    
    return {
        "task_id": task_id,
        "status": "started",
        "message": "Astronomical data download and processing started"
    }

@router.get("/astro/download/{task_id}")
async def get_download_status(task_id: str):
    """Get status of astronomical data download task."""
    if task_id not in background_tasks_status:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return background_tasks_status[task_id]

@router.get("/astro/status")
async def get_astronomical_status():
    """Get status of available astronomical catalogs with real data info."""
    from utils.astronomy_catalogs import get_catalog_info
    
    try:
        catalog_info = await get_catalog_info()
        return {
            "status": "ok",
            "catalogs": catalog_info,
            "data_directory": "galaxy_data/processed",
            "total_objects": sum(cat.get("rows", 0) for cat in catalog_info if cat.get("available")),
            "last_updated": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting catalog status: {e}")
        return {
            "status": "error",
            "message": str(e),
            "catalogs": []
        }

@router.get("/astro/statistics")
async def get_astronomical_statistics():
    """Get comprehensive statistics from real astronomical data."""
    try:
        from utils.astronomy_catalogs import get_comprehensive_statistics
        
        stats = await get_comprehensive_statistics()
        return {
            "status": "ok",
            **stats
        }
    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/astro/galaxies")
async def get_galaxies_data(
    source: Optional[str] = Query(None, description="Source catalog (SDSS, Euclid, DESI, DES)"),
    limit: int = Query(1000, ge=1, le=100000, description="Maximum number of rows"),
    min_z: Optional[float] = Query(None, description="Minimum redshift"),
    max_z: Optional[float] = Query(None, description="Maximum redshift"),
    min_ra: Optional[float] = Query(None, description="Minimum RA"),
    max_ra: Optional[float] = Query(None, description="Maximum RA"),
    min_dec: Optional[float] = Query(None, description="Minimum DEC"),
    max_dec: Optional[float] = Query(None, description="Maximum DEC"),
    include_ml_features: bool = Query(False, description="Include ML-ready features")
):
    """Get filtered galaxy data from real astronomical catalogs."""
    try:
        from utils.astronomy_catalogs import fetch_filtered_galaxies
        
        filters = {
            "source": source,
            "limit": limit,
            "min_z": min_z,
            "max_z": max_z,
            "min_ra": min_ra,
            "max_ra": max_ra,
            "min_dec": min_dec,
            "max_dec": max_dec
        }
        
        data = await fetch_filtered_galaxies(filters, include_ml_features)
        
        return {
            "status": "ok",
            "count": len(data["galaxies"]),
            "galaxies": data["galaxies"],
            "filters_applied": filters,
            "ml_features_included": include_ml_features,
            "processing_time": data.get("processing_time", 0)
        }
    except Exception as e:
        logger.error(f"Error fetching galaxy data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ml/prepare-dataset")
async def prepare_ml_dataset(
    background_tasks: BackgroundTasks,
    target_variable: str = Query("redshift", description="Target variable for ML"),
    test_size: float = Query(0.2, ge=0.1, le=0.5, description="Test set size"),
    include_features: List[str] = Query(None, description="Features to include"),
    normalization: str = Query("standard", description="Normalization method")
):
    """Prepare ML-ready dataset from astronomical data."""
    import uuid
    task_id = str(uuid.uuid4())
    
    config = {
        "target_variable": target_variable,
        "test_size": test_size,
        "include_features": include_features,
        "normalization": normalization
    }
    
    background_tasks.add_task(prepare_ml_data, task_id, config)
    background_tasks_status[task_id] = {
        "status": "started",
        "progress": 0,
        "message": "Preparing ML dataset",
        "config": config,
        "started_at": datetime.now().isoformat()
    }
    
    return {
        "task_id": task_id,
        "status": "started",
        "message": "ML dataset preparation started",
        "config": config
    }

@router.get("/ml/dataset/{task_id}")
async def get_ml_dataset_status(task_id: str):
    """Get status of ML dataset preparation."""
    if task_id not in background_tasks_status:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return background_tasks_status[task_id]

@router.get("/ml/dataset/{task_id}/download")
async def download_ml_dataset(task_id: str):
    """Download prepared ML dataset."""
    if task_id not in background_tasks_status:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task_status = background_tasks_status[task_id]
    if task_status["status"] != "completed":
        raise HTTPException(status_code=400, detail="Dataset not ready")
    
    if "download_path" not in task_status:
        raise HTTPException(status_code=404, detail="Dataset file not found")
    
    file_path = task_status["download_path"]
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Dataset file not found on disk")
    
    return FileResponse(
        path=file_path,
        filename=f"ml_dataset_{task_id}.zip",
        media_type="application/zip"
    )

@router.get("/ads/search")
async def search_ads_publications(
    query: str = Query(..., description="Search query"),
    search_type: str = Query("general", description="Search type: general, coordinates, object, catalog"),
    max_results: int = Query(20, ge=1, le=100, description="Maximum results"),
    ra: Optional[float] = Query(None, description="RA for coordinate search"),
    dec: Optional[float] = Query(None, description="DEC for coordinate search"),
    radius: Optional[float] = Query(0.1, description="Search radius in degrees")
):
    """Search NASA ADS for astronomical publications."""
    try:
        from utils.ads_astronomy import search_ads_advanced
        
        search_params = {
            "query": query,
            "search_type": search_type,
            "max_results": max_results,
            "ra": ra,
            "dec": dec,
            "radius": radius
        }
        
        results = await search_ads_advanced(search_params)
        
        return {
            "status": "ok",
            "count": len(results["publications"]),
            "publications": results["publications"],
            "search_params": search_params,
            "total_found": results.get("total_found", len(results["publications"]))
        }
    except Exception as e:
        logger.error(f"Error searching ADS: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/data/process")
async def process_custom_data(
    background_tasks: BackgroundTasks,
    processing_type: str = Query(..., description="Processing type: clean, normalize, feature_engineering"),
    input_format: str = Query("csv", description="Input format"),
    output_format: str = Query("csv", description="Output format")
):
    """Process custom astronomical data with cleaning and normalization."""
    import uuid
    task_id = str(uuid.uuid4())
    
    config = {
        "processing_type": processing_type,
        "input_format": input_format,
        "output_format": output_format
    }
    
    background_tasks.add_task(process_custom_astronomical_data, task_id, config)
    background_tasks_status[task_id] = {
        "status": "started",
        "progress": 0,
        "message": "Processing custom data",
        "config": config,
        "started_at": datetime.now().isoformat()
    }
    
    return {
        "task_id": task_id,
        "status": "started",
        "message": "Custom data processing started",
        "config": config
    }

# Background task functions
async def process_astronomical_data(task_id: str):
    """Background task to download and process real astronomical data."""
    try:
        background_tasks_status[task_id]["message"] = "Importing required modules"
        background_tasks_status[task_id]["progress"] = 5
        
        from utils.astronomy_catalogs import (
            get_sdss_data, get_euclid_data, get_desi_data, 
            get_des_data, merge_all_data, convert_to_cartesian
        )
        
        catalogs = []
        progress_step = 20
        
        # Download SDSS data
        background_tasks_status[task_id]["message"] = "Downloading SDSS DR17 data"
        background_tasks_status[task_id]["progress"] = 10
        try:
            sdss_path = await asyncio.to_thread(get_sdss_data)
            catalogs.append({"name": "SDSS DR17", "path": sdss_path, "status": "success"})
            background_tasks_status[task_id]["progress"] += progress_step
        except Exception as e:
            catalogs.append({"name": "SDSS DR17", "status": "error", "error": str(e)})
        
        # Download Euclid data
        background_tasks_status[task_id]["message"] = "Downloading Euclid Q1 data"
        try:
            euclid_path = await asyncio.to_thread(get_euclid_data)
            catalogs.append({"name": "Euclid Q1", "path": euclid_path, "status": "success"})
            background_tasks_status[task_id]["progress"] += progress_step
        except Exception as e:
            catalogs.append({"name": "Euclid Q1", "status": "error", "error": str(e)})
        
        # Download DESI data
        background_tasks_status[task_id]["message"] = "Downloading DESI DR1 data"
        try:
            desi_path = await asyncio.to_thread(get_desi_data)
            catalogs.append({"name": "DESI DR1", "path": desi_path, "status": "success"})
            background_tasks_status[task_id]["progress"] += progress_step
        except Exception as e:
            catalogs.append({"name": "DESI DR1", "status": "error", "error": str(e)})
        
        # Download DES data
        background_tasks_status[task_id]["message"] = "Downloading DES Y6 data"
        try:
            des_path = await asyncio.to_thread(get_des_data)
            catalogs.append({"name": "DES Y6", "path": des_path, "status": "success"})
            background_tasks_status[task_id]["progress"] += progress_step
        except Exception as e:
            catalogs.append({"name": "DES Y6", "status": "error", "error": str(e)})
        
        # Merge data
        background_tasks_status[task_id]["message"] = "Merging datasets"
        try:
            merged_path = await asyncio.to_thread(merge_all_data)
            catalogs.append({"name": "Merged Dataset", "path": merged_path, "status": "success"})
            background_tasks_status[task_id]["progress"] = 95
        except Exception as e:
            catalogs.append({"name": "Merged Dataset", "status": "error", "error": str(e)})
        
        # Complete
        background_tasks_status[task_id] = {
            "status": "completed",
            "progress": 100,
            "message": "All datasets processed successfully",
            "catalogs": catalogs,
            "completed_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        background_tasks_status[task_id] = {
            "status": "failed",
            "error": str(e),
            "failed_at": datetime.now().isoformat()
        }

async def prepare_ml_data(task_id: str, config: dict):
    """Background task to prepare ML-ready dataset."""
    try:
        background_tasks_status[task_id]["message"] = "Loading astronomical data"
        background_tasks_status[task_id]["progress"] = 10
        
        from utils.ml_preprocessing import prepare_ml_dataset
        
        result = await asyncio.to_thread(prepare_ml_dataset, config)
        
        background_tasks_status[task_id] = {
            "status": "completed",
            "progress": 100,
            "message": "ML dataset prepared successfully",
            "result": result,
            "download_path": result.get("file_path"),
            "completed_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        background_tasks_status[task_id] = {
            "status": "failed",
            "error": str(e),
            "failed_at": datetime.now().isoformat()
        }

async def process_custom_astronomical_data(task_id: str, config: dict):
    """Background task to process custom astronomical data."""
    try:
        background_tasks_status[task_id]["message"] = "Processing custom data"
        background_tasks_status[task_id]["progress"] = 20
        
        from utils.data_processing import process_astronomical_data
        
        result = await asyncio.to_thread(process_astronomical_data, config)
        
        background_tasks_status[task_id] = {
            "status": "completed",
            "progress": 100,
            "message": "Custom data processed successfully",
            "result": result,
            "completed_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        background_tasks_status[task_id] = {
            "status": "failed",
            "error": str(e),
            "failed_at": datetime.now().isoformat()
        }

# Include the router in the app
app.include_router(router) 