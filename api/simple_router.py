"""
Simple Router module for basic API functionality.
This module doesn't depend on pandas or other heavy libraries.
"""

from fastapi import APIRouter, Query
from typing import Dict, Any, List, Optional

router = APIRouter()

@router.get("/status")
async def get_status():
    """
    Return simple status information for the API.
    """
    return {
        "status": "ok",
        "catalogs": [
            {"name": "SDSS DR17", "description": "Spectroscopic catalog", "available": False},
            {"name": "Euclid Q1", "description": "MER Final catalog", "available": False},
            {"name": "DESI DR1", "description": "ELG clustering catalog", "available": False},
            {"name": "DES Y6", "description": "Gold catalog", "available": False}
        ],
        "message": "This is a simplified version of the API. For full functionality, ensure all dependencies are installed."
    }

@router.get("/galaxies")
async def get_galaxies(
    source: Optional[str] = Query(None, description="Data source"),
    limit: int = Query(10, ge=1, le=100, description="Maximum number of rows to return"),
):
    """
    Return sample galaxy data.
    """
    # Sample data that doesn't require pandas
    sample_galaxies = [
        {"RA": 150.1, "DEC": 2.2, "redshift": 0.5, "source": "SDSS", "X": 100.0, "Y": 200.0, "Z": 300.0},
        {"RA": 151.2, "DEC": 2.3, "redshift": 0.6, "source": "SDSS", "X": 110.0, "Y": 210.0, "Z": 310.0},
        {"RA": 149.3, "DEC": 2.1, "redshift": 0.4, "source": "Euclid", "X": 90.0, "Y": 190.0, "Z": 290.0},
        {"RA": 152.4, "DEC": 2.4, "redshift": 0.7, "source": "DESI", "X": 120.0, "Y": 220.0, "Z": 320.0},
        {"RA": 153.5, "DEC": 2.5, "redshift": 0.8, "source": "DESI", "X": 130.0, "Y": 230.0, "Z": 330.0},
        {"RA": 148.6, "DEC": 2.0, "redshift": 0.3, "source": "DES", "X": 80.0, "Y": 180.0, "Z": 280.0},
        {"RA": 147.7, "DEC": 1.9, "redshift": 0.2, "source": "DES", "X": 70.0, "Y": 170.0, "Z": 270.0},
        {"RA": 146.8, "DEC": 1.8, "redshift": 0.1, "source": "DES", "X": 60.0, "Y": 160.0, "Z": 260.0},
        {"RA": 154.9, "DEC": 2.6, "redshift": 0.9, "source": "Euclid", "X": 140.0, "Y": 240.0, "Z": 340.0},
        {"RA": 156.0, "DEC": 2.7, "redshift": 1.0, "source": "Euclid", "X": 150.0, "Y": 250.0, "Z": 350.0},
    ]
    
    # Apply source filter if specified
    if source:
        filtered_galaxies = [g for g in sample_galaxies if g["source"] == source]
    else:
        filtered_galaxies = sample_galaxies
    
    # Apply limit
    result = filtered_galaxies[:limit]
    
    return {
        "count": len(result),
        "galaxies": result
    }

@router.get("/statistics")
async def get_statistics():
    """
    Return sample statistics for the API.
    """
    return {
        "total_galaxies": 10000,
        "redshift": {
            "min": 0.1,
            "max": 1.0,
            "mean": 0.55
        },
        "sources": {
            "SDSS": 2500,
            "Euclid": 3000,
            "DESI": 2000,
            "DES": 2500
        }
    } 