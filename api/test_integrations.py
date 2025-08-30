"""
API endpoint for testing all data source integrations
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List
import os
import logging

# Import external API clients
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/test", tags=["testing"])

@router.get("/integrations")
async def test_all_integrations():
    """Test connectivity to all external data sources"""
    results = {}
    
    # Test ADS API
    try:
        ads_token = os.getenv("ADSABS_TOKEN")
        if ads_token:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "https://api.adsabs.harvard.edu/v1/search/query",
                    params={"q": "astronomy", "rows": 1},
                    headers={"Authorization": f"Bearer {ads_token}"},
                    timeout=10.0
                )
                results["ads"] = {
                    "status": "ok" if response.status_code == 200 else "error",
                    "status_code": response.status_code,
                    "message": "ADS API accessible" if response.status_code == 200 else response.text[:100]
                }
        else:
            results["ads"] = {"status": "not_configured", "message": "ADSABS_TOKEN not set"}
    except Exception as e:
        results["ads"] = {"status": "error", "message": str(e)}
    
    # Test arXiv API
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://export.arxiv.org/api/query",
                params={"search_query": "cat:astro-ph", "max_results": 1},
                timeout=10.0
            )
            results["arxiv"] = {
                "status": "ok" if response.status_code == 200 else "error",
                "status_code": response.status_code,
                "message": "arXiv API accessible" if response.status_code == 200 else response.text[:100]
            }
    except Exception as e:
        results["arxiv"] = {"status": "error", "message": str(e)}
    
    # Test SerpAPI
    try:
        serpapi_key = os.getenv("SERPAPI_KEY")
        if serpapi_key:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "https://serpapi.com/search.json",
                    params={
                        "engine": "google_scholar",
                        "q": "astronomy",
                        "num": 1,
                        "api_key": serpapi_key
                    },
                    timeout=10.0
                )
                results["serpapi"] = {
                    "status": "ok" if response.status_code == 200 else "error",
                    "status_code": response.status_code,
                    "message": "SerpAPI accessible" if response.status_code == 200 else response.text[:100]
                }
        else:
            results["serpapi"] = {"status": "not_configured", "message": "SERPAPI_KEY not set"}
    except Exception as e:
        results["serpapi"] = {"status": "error", "message": str(e)}
    
    # Test NASA API
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://exoplanetarchive.ipac.caltech.edu/TAP/sync",
                params={
                    "query": "SELECT TOP 1 pl_name FROM pscomppars WHERE default_flag = 1",
                    "format": "json"
                },
                timeout=10.0
            )
            results["nasa"] = {
                "status": "ok" if response.status_code == 200 else "error",
                "status_code": response.status_code,
                "message": "NASA Exoplanet Archive accessible" if response.status_code == 200 else response.text[:100]
            }
    except Exception as e:
        results["nasa"] = {"status": "error", "message": str(e)}
    
    # Test SDSS API
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://skyserver.sdss.org/dr17/SearchTools/sql",
                params={
                    "cmd": "SELECT TOP 1 ra, dec FROM SpecObj WHERE class = 'galaxy'",
                    "format": "json"
                },
                timeout=10.0
            )
            results["sdss"] = {
                "status": "ok" if response.status_code == 200 else "error",
                "status_code": response.status_code,
                "message": "SDSS SkyServer accessible" if response.status_code == 200 else response.text[:100]
            }
    except Exception as e:
        results["sdss"] = {"status": "error", "message": str(e)}
    
    # Calculate overall status
    total_sources = len(results)
    working_sources = sum(1 for r in results.values() if r["status"] == "ok")
    configured_sources = sum(1 for r in results.values() if r["status"] in ["ok", "error"])
    
    return {
        "overall_status": "healthy" if working_sources >= total_sources * 0.7 else "partial" if working_sources > 0 else "down",
        "working_sources": working_sources,
        "total_sources": total_sources,
        "configured_sources": configured_sources,
        "sources": results,
        "recommendations": get_integration_recommendations(results)
    }

def get_integration_recommendations(results: Dict) -> List[str]:
    """Get recommendations for improving integrations"""
    recommendations = []
    
    for source, result in results.items():
        if result["status"] == "not_configured":
            recommendations.append(f"Configure {source.upper()} API key for enhanced data collection")
        elif result["status"] == "error":
            recommendations.append(f"Check {source.upper()} API connectivity and credentials")
    
    if not recommendations:
        recommendations.append("All configured integrations are working properly!")
    
    return recommendations

@router.get("/templates")
async def get_available_templates():
    """Get list of available import templates with their configurations"""
    from api.data_management import IMPORT_TEMPLATES
    
    # Add status for each template based on API availability
    templates_with_status = {}
    
    for template_id, template in IMPORT_TEMPLATES.items():
        source = template["source"]
        
        # Check if required credentials are available
        if source == "ads":
            available = bool(os.getenv("ADSABS_TOKEN"))
        elif source == "serpapi":
            available = bool(os.getenv("SERPAPI_KEY"))
        elif source in ["arxiv", "nasa", "sdss"]:
            available = True  # These don't require API keys
        else:
            available = False
        
        templates_with_status[template_id] = {
            **template,
            "available": available,
            "requires_auth": source in ["ads", "serpapi"]
        }
    
    return {
        "templates": templates_with_status,
        "total_templates": len(templates_with_status),
        "available_templates": sum(1 for t in templates_with_status.values() if t["available"])
    }
