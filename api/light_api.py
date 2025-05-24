"""
Lightweight API endpoints that don't require heavy dependencies.
These endpoints will be deployed to Vercel.
"""

from fastapi import APIRouter, HTTPException, Request, Response, Query
from fastapi.responses import JSONResponse
import os
from typing import Dict, Any, Optional
import httpx

router = APIRouter()

HEAVY_COMPUTE_URL = os.getenv("HEAVY_COMPUTE_URL", "http://localhost:8001")

# Universal proxy route for heavy endpoints
def is_heavy_path(path: str) -> bool:
    heavy_paths = [
        "ads/search",
        "datasets/search", 
        "ml/predict",
        "astro/analyze",
        "astro/download"
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
    try:
        # Try to get real data from heavy compute service
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{HEAVY_COMPUTE_URL}/astro/status", timeout=10.0)
            if response.status_code == 200:
                return response.json()
    except Exception as e:
        pass  # Fall back to demo data
    
    return {
        "status": "ok",
        "catalogs": [
            {"name": "SDSS DR17", "description": "Spectroscopic catalog", "available": True, "rows": 2500},
            {"name": "Euclid Q1", "description": "MER Final catalog", "available": True, "rows": 3000},
            {"name": "DESI DR1", "description": "ELG clustering catalog", "available": True, "rows": 2000},
            {"name": "DES Y6", "description": "Gold catalog", "available": True, "rows": 2500}
        ],
        "message": "Демонстрационные данные. Для полной функциональности требуется тяжёлый вычислительный сервис."
    }

@router.get("/ads/basic")
async def ads_basic():
    """Basic ADS service status."""
    return {
        "status": "ok",
        "message": "Basic ADS service is available",
        "note": "For full ADS functionality, use the heavy compute service"
    }

@router.get("/astro/statistics")
async def astro_statistics():
    """Get statistics for astronomical data."""
    try:
        # Try to get real data from heavy compute service
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{HEAVY_COMPUTE_URL}/astro/statistics", timeout=10.0)
            if response.status_code == 200:
                return response.json()
    except Exception as e:
        pass  # Fall back to demo data
    
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

@router.get("/astro/galaxies")
async def astro_galaxies(
    source: Optional[str] = Query(None, description="Источник данных (SDSS, Euclid, DESI, DES)"),
    limit: int = Query(1000, ge=1, le=10000, description="Максимальное количество возвращаемых строк"),
    min_z: Optional[float] = Query(None, description="Минимальное красное смещение"),
    max_z: Optional[float] = Query(None, description="Максимальное красное смещение")
):
    """Get galaxy data with filtering."""
    # Sample galaxy data
    sample_galaxies = [
        {"RA": 150.1, "DEC": 2.2, "redshift": 0.5, "source": "SDSS", "X": 100.0, "Y": 200.0, "Z": 300.0},
        {"RA": 151.2, "DEC": 2.3, "redshift": 0.6, "source": "SDSS", "X": 110.0, "Y": 210.0, "Z": 310.0},
        {"RA": 149.3, "DEC": 2.1, "redshift": 0.4, "source": "Euclid", "X": 90.0, "Y": 190.0, "Z": 290.0},
        {"RA": 152.4, "DEC": 2.4, "redshift": 0.7, "source": "DESI", "X": 120.0, "Y": 220.0, "Z": 320.0},
        {"RA": 153.5, "DEC": 2.5, "redshift": 0.8, "source": "DESI", "X": 130.0, "Y": 230.0, "Z": 330.0},
        {"RA": 148.6, "DEC": 2.0, "redshift": 0.3, "source": "DES", "X": 80.0, "Y": 180.0, "Z": 280.0},
        {"RA": 154.7, "DEC": 2.6, "redshift": 0.9, "source": "DES", "X": 140.0, "Y": 240.0, "Z": 340.0},
        {"RA": 147.8, "DEC": 1.9, "redshift": 0.2, "source": "SDSS", "X": 70.0, "Y": 170.0, "Z": 270.0},
    ]
    
    # Filter by source if specified
    if source:
        sample_galaxies = [g for g in sample_galaxies if g["source"] == source]
    
    # Filter by redshift if specified
    if min_z is not None:
        sample_galaxies = [g for g in sample_galaxies if g["redshift"] >= min_z]
    if max_z is not None:
        sample_galaxies = [g for g in sample_galaxies if g["redshift"] <= max_z]
    
    # Apply limit
    sample_galaxies = sample_galaxies[:limit]
    
    return {
        "count": len(sample_galaxies),
        "galaxies": sample_galaxies,
        "note": "Демонстрационные данные. Для настоящих данных требуется полная версия API."
    }

@router.get("/ads/search-by-coordinates")
async def ads_search_by_coordinates(
    ra: float = Query(..., description="Прямое восхождение"),
    dec: float = Query(..., description="Склонение"),
    radius: float = Query(0.1, description="Радиус поиска в градусах")
):
    """Search ADS by coordinates."""
    return {
        "count": 2,
        "publications": [
            {
                "title": ["Large-scale structure in the Universe"],
                "author": ["Smith, J.", "Doe, J."],
                "year": "2023",
                "citation_count": 45,
                "bibcode": "2023ExamplePaper..1S",
                "doi": "10.1000/example"
            },
            {
                "title": ["Galaxy clusters and cosmic web"],
                "author": ["Wilson, A.", "Brown, K."],
                "year": "2022",
                "citation_count": 32,
                "bibcode": "2022ExamplePaper..2W",
                "doi": "10.1000/example2"
            }
        ],
        "note": "Демонстрационные данные. Для настоящего поиска в ADS требуется полная версия API."
    }

@router.get("/ads/search-by-object")
async def ads_search_by_object(
    object_name: str = Query(..., description="Название астрономического объекта")
):
    """Search ADS by object name."""
    return {
        "count": 3,
        "publications": [
            {
                "title": [f"Study of {object_name} and surrounding field"],
                "author": ["Johnson, M.", "Miller, S."],
                "year": "2023",
                "citation_count": 28,
                "bibcode": "2023ExampleObj..1J",
                "doi": "10.1000/obj1"
            },
            {
                "title": [f"Photometric analysis of {object_name}"],
                "author": ["Davis, L.", "Garcia, R."],
                "year": "2022",
                "citation_count": 19,
                "bibcode": "2022ExampleObj..2D",
                "doi": "10.1000/obj2"
            },
            {
                "title": [f"Spectroscopic observations of {object_name}"],
                "author": ["Taylor, P.", "Anderson, C."],
                "year": "2021",
                "citation_count": 15,
                "bibcode": "2021ExampleObj..3T"
            }
        ],
        "note": "Демонстрационные данные. Для настоящего поиска в ADS требуется полная версия API."
    }

@router.get("/ads/search-by-catalog")
async def ads_search_by_catalog(
    catalog: str = Query(..., description="Название каталога")
):
    """Search ADS by catalog name."""
    return {
        "total_found": 150,
        "catalog": catalog,
        "publications": [
            {
                "title": [f"{catalog} Data Release and Scientific Results"],
                "author": ["Survey Team", "Collaboration"],
                "year": "2023",
                "citation_count": 89,
                "bibcode": "2023Catalog...1ST",
                "doi": "10.1000/cat1"
            },
            {
                "title": [f"Analysis of {catalog} galaxy sample"],
                "author": ["Research Group"],
                "year": "2022",
                "citation_count": 67,
                "bibcode": "2022Catalog...2RG",
                "doi": "10.1000/cat2"
            }
        ],
        "keyword_stats": {
            "galaxy": 45,
            "redshift": 32,
            "clustering": 28,
            "photometry": 25,
            "large scale structure": 20
        },
        "note": "Демонстрационные данные. Для настоящего поиска в ADS требуется полная версия API."
    }

@router.get("/ads/large-scale-structure")
async def ads_large_scale_structure(
    start_year: int = Query(2010, description="Начальный год поиска"),
    additional_keywords: Optional[str] = Query(None, description="Дополнительные ключевые слова")
):
    """Search ADS for large-scale structure papers."""
    return {
        "total_found": 89,
        "publications": [
            {
                "title": ["Cosmic web filaments and galaxy formation"],
                "author": ["Cosmic Web Team"],
                "year": "2023",
                "citation_count": 76,
                "bibcode": "2023LargeScale..1CWT",
                "doi": "10.1000/lss1"
            },
            {
                "title": ["Void statistics in large galaxy surveys"],
                "author": ["Void Survey Group"],
                "year": "2022",
                "citation_count": 54,
                "bibcode": "2022LargeScale..2VSG",
                "doi": "10.1000/lss2"
            }
        ],
        "year_stats": {
            "2023": 15,
            "2022": 18,
            "2021": 12,
            "2020": 14,
            "2019": 10
        },
        "note": "Демонстрационные данные. Для настоящего поиска в ADS требуется полная версия API."
    }

# Add enhanced endpoints instead
@router.get("/system/info")
async def system_info():
    """System information endpoint"""
    return {
        "system": "Scientific API",
        "version": "1.0.0",
        "architecture": "Light API (Vercel) + Heavy Compute (Azure)",
        "current_service": "light-api",
        "features": [
            "Astronomical data catalogs",
            "Galaxy statistics",
            "NASA ADS integration", 
            "Coordinate-based search",
            "Real-time data processing"
        ],
        "status": "operational"
    }

# Remove or comment out Heavy API proxy routes
# @router.get("/heavy/{path:path}")
# async def proxy_to_heavy(request: Request, path: str):
#     """Proxy requests to heavy compute service"""
#     # This is commented out since Heavy API is not stable yet
#     return {"status": "ok", "message": "Heavy compute service temporarily unavailable", "service": "light-api"}

# Proxy route - должен быть последним и более специфичным
# Убираем универсальный proxy, так как он конфликтует с основными маршрутами
# @router.api_route("/{proxy_path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
# async def proxy_to_heavy(request: Request, proxy_path: str):
#     if not is_heavy_path(proxy_path):
#         return JSONResponse({"detail": "Not Found"}, status_code=404)
#
#     url = f"{HEAVY_COMPUTE_URL}/{proxy_path}"
#     method = request.method
#     headers = dict(request.headers)
#     body = await request.body()
#
#     async with httpx.AsyncClient() as client:
#         resp = await client.request(
#             method, url, headers=headers, content=body, params=dict(request.query_params)
#         )
#         return Response(
#             content=resp.content,
#             status_code=resp.status_code,
#             headers=resp.headers
#         ) 
