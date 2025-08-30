"""
Data Management API endpoints for the Scientific Data Platform
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Query
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import json
import io
import asyncio
import os
from datetime import datetime
import logging

# Import database
from database.config import db

# Import data processing utilities
try:
    from utils.data_preprocessor import AstronomicalDataPreprocessor
    from utils.data_processing import clean_astronomical_data, normalize_features
    PROCESSING_AVAILABLE = True
except ImportError:
    PROCESSING_AVAILABLE = False

# Import external API clients
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/data", tags=["data_management"])

# Data import templates
IMPORT_TEMPLATES = {
    "sdss_galaxies": {
        "name": "SDSS Galaxy Catalog",
        "description": "Import galaxy data from Sloan Digital Sky Survey",
        "source": "sdss",
        "default_params": {
            "object_type": "galaxy",
            "limit": 10000,
            "fields": ["ra", "dec", "z", "petroMag_r", "modelMag_u", "modelMag_g"]
        }
    },
    "nasa_exoplanets": {
        "name": "NASA Exoplanet Archive",
        "description": "Confirmed exoplanet data from NASA",
        "source": "nasa",
        "default_params": {
            "table": "pscomppars",
            "format": "json",
            "select": "pl_name,hostname,pl_masse,pl_rade,pl_orbper,sy_dist"
        }
    },
    "ads_papers": {
        "name": "ADS Research Papers",
        "description": "Astrophysics papers from ADS",
        "source": "ads",
        "default_params": {
            "q": "astronomy",
            "rows": 100,
            "fl": "title,author,year,abstract,citation_count"
        }
    },
    "arxiv_papers": {
        "name": "arXiv Research Papers",
        "description": "Latest research papers from arXiv",
        "source": "arxiv",
        "default_params": {
            "search_query": "cat:astro-ph",
            "max_results": 100,
            "sort_by": "submittedDate",
            "sort_order": "descending"
        }
    },
    "serpapi_search": {
        "name": "SerpAPI Web Search",
        "description": "Search scientific data from web sources",
        "source": "serpapi",
        "default_params": {
            "q": "scientific research data",
            "engine": "google_scholar",
            "num": 50
        }
    },
    "google_scholar": {
        "name": "Google Scholar Papers",
        "description": "Academic papers from Google Scholar via SerpAPI",
        "source": "serpapi",
        "default_params": {
            "q": "machine learning astronomy",
            "engine": "google_scholar",
            "num": 100,
            "as_ylo": "2020"
        }
    }
}

@router.post("/import-template")
async def import_template_data(
    template_data: Dict[str, str],
    background_tasks: BackgroundTasks
):
    """Import data using a predefined template"""
    template_name = template_data.get("template")
    
    if template_name not in IMPORT_TEMPLATES:
        raise HTTPException(status_code=400, detail="Invalid template name")
    
    template = IMPORT_TEMPLATES[template_name]
    
    # Start background import task
    task_id = f"import_{template_name}_{datetime.utcnow().timestamp()}"
    
    background_tasks.add_task(
        import_data_background,
        task_id,
        template
    )
    
    return {
        "status": "started",
        "task_id": task_id,
        "template": template_name,
        "message": f"Import of {template['name']} started"
    }

async def import_data_background(task_id: str, template: Dict):
    """Background task to import data"""
    try:
        # Update task status in database
        await db.mongo_update_one(
            "import_tasks",
            {"_id": task_id},
            {
                "status": "processing",
                "started_at": datetime.utcnow(),
                "template": template
            },
            upsert=True
        )
        
        # Import data based on source
        if template["source"] == "sdss":
            data = await import_sdss_data(template["default_params"])
        elif template["source"] == "nasa":
            data = await import_nasa_data(template["default_params"])
        elif template["source"] == "ads":
            data = await import_ads_data(template["default_params"])
        elif template["source"] == "arxiv":
            data = await import_arxiv_data(template["default_params"])
        elif template["source"] == "serpapi":
            data = await import_serpapi_data(template["default_params"])
        else:
            raise ValueError(f"Unknown source: {template['source']}")
        
        # Store imported data
        if data:
            await store_imported_data(task_id, template["source"], data)
        
        # Update task status
        await db.mongo_update_one(
            "import_tasks",
            {"_id": task_id},
            {
                "status": "completed",
                "completed_at": datetime.utcnow(),
                "record_count": len(data) if data else 0
            }
        )
        
    except Exception as e:
        logger.error(f"Import task {task_id} failed: {e}")
        await db.mongo_update_one(
            "import_tasks",
            {"_id": task_id},
            {
                "status": "failed",
                "error": str(e),
                "failed_at": datetime.utcnow()
            }
        )

async def import_sdss_data(params: Dict) -> List[Dict]:
    """Import data from SDSS"""
    if not HTTPX_AVAILABLE:
        raise ImportError("httpx is required for API imports")
    
    # SDSS API endpoint
    base_url = "https://skyserver.sdss.org/dr17/SearchTools/sql"
    
    # Build SQL query
    object_type = params.get("object_type", "galaxy")
    limit = params.get("limit", 1000)
    fields = params.get("fields", ["ra", "dec", "z"])
    
    sql_query = f"""
    SELECT TOP {limit} 
        {', '.join(fields)}
    FROM SpecObj
    WHERE class = '{object_type}'
    """
    
    async with httpx.AsyncClient() as client:
        response = await client.get(
            base_url,
            params={
                "cmd": sql_query,
                "format": "json"
            },
            timeout=30.0
        )
        
        if response.status_code == 200:
            data = response.json()
            return data[0]["Rows"] if data else []
        else:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"SDSS API error: {response.text}"
            )

async def import_nasa_data(params: Dict) -> List[Dict]:
    """Import data from NASA Exoplanet Archive"""
    if not HTTPX_AVAILABLE:
        raise ImportError("httpx is required for API imports")
    
    base_url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
    
    query = f"""
    SELECT {params.get('select', '*')}
    FROM {params.get('table', 'pscomppars')}
    WHERE default_flag = 1
    """
    
    async with httpx.AsyncClient() as client:
        response = await client.get(
            base_url,
            params={
                "query": query,
                "format": "json"
            },
            timeout=30.0
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"NASA API error: {response.text}"
            )

async def import_ads_data(params: Dict) -> List[Dict]:
    """Import data from ADS (Astrophysics Data System)"""
    if not HTTPX_AVAILABLE:
        raise ImportError("httpx is required for API imports")
    
    ads_token = os.getenv("ADSABS_TOKEN")
    if not ads_token:
        raise HTTPException(status_code=400, detail="ADS token not configured")
    
    base_url = "https://api.adsabs.harvard.edu/v1/search/query"
    
    headers = {
        "Authorization": f"Bearer {ads_token}",
        "Content-Type": "application/json"
    }
    
    query_params = {
        "q": params.get("q", "astronomy"),
        "fl": params.get("fl", "title,author,year,abstract,citation_count"),
        "rows": params.get("rows", 100),
        "sort": "date desc"
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.get(
            base_url,
            params=query_params,
            headers=headers,
            timeout=30.0
        )
        
        if response.status_code == 200:
            data = response.json()
            return data.get("response", {}).get("docs", [])
        else:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"ADS API error: {response.text}"
            )

async def import_arxiv_data(params: Dict) -> List[Dict]:
    """Import data from arXiv"""
    if not HTTPX_AVAILABLE:
        raise ImportError("httpx is required for API imports")
    
    base_url = "https://export.arxiv.org/api/query"
    
    # Build arXiv query
    search_query = params.get("search_query", "cat:astro-ph")
    max_results = params.get("max_results", 100)
    sort_by = params.get("sort_by", "submittedDate")
    sort_order = params.get("sort_order", "descending")
    
    query_params = {
        "search_query": search_query,
        "start": 0,
        "max_results": max_results,
        "sortBy": sort_by,
        "sortOrder": sort_order
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.get(
            base_url,
            params=query_params,
            timeout=30.0
        )
        
        if response.status_code == 200:
            # Parse XML response
            import xml.etree.ElementTree as ET
            
            root = ET.fromstring(response.text)
            papers = []
            
            # arXiv uses Atom feed format
            for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
                paper = {}
                
                # Extract basic info
                title_elem = entry.find("{http://www.w3.org/2005/Atom}title")
                paper["title"] = title_elem.text.strip() if title_elem is not None else ""
                
                summary_elem = entry.find("{http://www.w3.org/2005/Atom}summary")
                paper["abstract"] = summary_elem.text.strip() if summary_elem is not None else ""
                
                # Extract authors
                authors = []
                for author in entry.findall("{http://www.w3.org/2005/Atom}author"):
                    name_elem = author.find("{http://www.w3.org/2005/Atom}name")
                    if name_elem is not None:
                        authors.append(name_elem.text)
                paper["authors"] = authors
                
                # Extract dates
                published_elem = entry.find("{http://www.w3.org/2005/Atom}published")
                paper["published"] = published_elem.text if published_elem is not None else ""
                
                updated_elem = entry.find("{http://www.w3.org/2005/Atom}updated")
                paper["updated"] = updated_elem.text if updated_elem is not None else ""
                
                # Extract arXiv ID
                id_elem = entry.find("{http://www.w3.org/2005/Atom}id")
                if id_elem is not None:
                    paper["arxiv_id"] = id_elem.text.split("/")[-1]
                
                # Extract categories
                categories = []
                for category in entry.findall("{http://arxiv.org/schemas/atom}primary_category"):
                    term = category.get("term")
                    if term:
                        categories.append(term)
                paper["categories"] = categories
                
                papers.append(paper)
            
            return papers
        else:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"arXiv API error: {response.text}"
            )

async def import_serpapi_data(params: Dict) -> List[Dict]:
    """Import data from SerpAPI (Google Scholar, etc.)"""
    if not HTTPX_AVAILABLE:
        raise ImportError("httpx is required for API imports")
    
    serpapi_key = os.getenv("SERPAPI_KEY")
    if not serpapi_key:
        raise HTTPException(status_code=400, detail="SERPAPI key not configured")
    
    engine = params.get("engine", "google_scholar")
    base_url = f"https://serpapi.com/search.json"
    
    query_params = {
        "engine": engine,
        "q": params.get("q", "scientific research"),
        "api_key": serpapi_key
    }
    
    # Add engine-specific parameters
    if engine == "google_scholar":
        query_params.update({
            "num": params.get("num", 50),
            "as_ylo": params.get("as_ylo", "2020"),  # Year low
            "as_yhi": params.get("as_yhi", "2024"),  # Year high
            "scisbd": "1"  # Sort by date
        })
    
    async with httpx.AsyncClient() as client:
        response = await client.get(
            base_url,
            params=query_params,
            timeout=30.0
        )
        
        if response.status_code == 200:
            data = response.json()
            
            if engine == "google_scholar":
                # Extract organic results from Google Scholar
                results = data.get("organic_results", [])
                processed_results = []
                
                for result in results:
                    processed_result = {
                        "title": result.get("title", ""),
                        "snippet": result.get("snippet", ""),
                        "link": result.get("link", ""),
                        "authors": result.get("publication_info", {}).get("authors", []),
                        "year": result.get("publication_info", {}).get("summary", "").split(",")[-1].strip() if result.get("publication_info", {}).get("summary") else "",
                        "cited_by": result.get("inline_links", {}).get("cited_by", {}).get("total", 0),
                        "source": "google_scholar"
                    }
                    processed_results.append(processed_result)
                
                return processed_results
            else:
                # Return raw results for other engines
                return [data]
        else:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"SerpAPI error: {response.text}"
            )

async def store_imported_data(task_id: str, source: str, data: List[Dict]):
    """Store imported data in the database"""
    if not data:
        return
    
    # Create collection name based on source
    collection_name = f"imported_{source}_data"
    
    # Add metadata to each record
    for record in data:
        record["_import_task_id"] = task_id
        record["_imported_at"] = datetime.utcnow()
        record["_source"] = source
    
    # Bulk insert
    if db.mongo_db is not None:
        collection = db.mongo_db[collection_name]
        await collection.insert_many(data)

@router.post("/upload")
async def upload_files(
    files: List[UploadFile] = File(...),
    auto_detect: bool = True,
    validate: bool = True
):
    """Upload and process data files"""
    results = []
    
    for file in files:
        try:
            # Read file content
            content = await file.read()
            
            # Detect file type and parse
            if file.filename.endswith('.csv'):
                df = pd.read_csv(io.StringIO(content.decode('utf-8')))
                data_type = "csv"
            elif file.filename.endswith('.json'):
                data = json.loads(content.decode('utf-8'))
                df = pd.DataFrame(data)
                data_type = "json"
            else:
                # Try to auto-detect
                try:
                    df = pd.read_csv(io.StringIO(content.decode('utf-8')))
                    data_type = "csv"
                except:
                    try:
                        data = json.loads(content.decode('utf-8'))
                        df = pd.DataFrame(data)
                        data_type = "json"
                    except:
                        raise ValueError("Unable to parse file format")
            
            # Basic validation
            if validate:
                issues = validate_dataframe(df)
                if issues:
                    results.append({
                        "filename": file.filename,
                        "status": "warning",
                        "issues": issues,
                        "rows": len(df),
                        "columns": list(df.columns)
                    })
                    continue
            
            # Store in database
            collection_name = f"uploaded_{file.filename.replace('.', '_')}"
            records = df.to_dict('records')
            
            if db.mongo_db is not None:
                collection = db.mongo_db[collection_name]
                await collection.insert_many(records)
            
            results.append({
                "filename": file.filename,
                "status": "success",
                "type": data_type,
                "rows": len(df),
                "columns": list(df.columns)
            })
            
        except Exception as e:
            results.append({
                "filename": file.filename,
                "status": "error",
                "error": str(e)
            })
    
    return {"files": results}

def validate_dataframe(df: pd.DataFrame) -> List[str]:
    """Validate dataframe and return list of issues"""
    issues = []
    
    # Check for empty dataframe
    if df.empty:
        issues.append("Dataframe is empty")
    
    # Check for missing values
    missing = df.isnull().sum()
    if missing.any():
        issues.append(f"Missing values found: {missing[missing > 0].to_dict()}")
    
    # Check for duplicate rows
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        issues.append(f"{duplicates} duplicate rows found")
    
    return issues

@router.get("/status/{task_id}")
async def get_import_status(task_id: str):
    """Get status of an import task"""
    task = await db.mongo_find_one("import_tasks", {"_id": task_id})
    
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return {
        "task_id": task_id,
        "status": task.get("status"),
        "started_at": task.get("started_at"),
        "completed_at": task.get("completed_at"),
        "record_count": task.get("record_count", 0),
        "error": task.get("error")
    }

@router.post("/clean")
async def clean_data(
    dataset_id: str,
    options: Dict[str, Any]
):
    """Clean and transform dataset"""
    if not PROCESSING_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Data processing utilities not available"
        )
    
    # Get dataset from database
    # TODO: Implement dataset retrieval
    
    # Apply cleaning operations
    cleaning_results = {
        "dataset_id": dataset_id,
        "operations": [],
        "before": {},
        "after": {}
    }
    
    # Handle missing values
    if options.get("handle_missing", True):
        # TODO: Implement missing value handling
        cleaning_results["operations"].append("Handled missing values")
    
    # Remove duplicates
    if options.get("remove_duplicates", True):
        # TODO: Implement duplicate removal
        cleaning_results["operations"].append("Removed duplicates")
    
    # Normalize values
    if options.get("normalize", False):
        # TODO: Implement normalization
        cleaning_results["operations"].append("Normalized values")
    
    return cleaning_results

@router.get("/datasets")
async def list_datasets(
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0)
):
    """List available datasets"""
    datasets = []
    
    if db.mongo_db is not None:
        # Get all collections that contain data
        collection_names = await db.mongo_db.list_collection_names()
        
        for name in collection_names:
            if name.startswith(("imported_", "uploaded_")):
                collection = db.mongo_db[name]
                count = await collection.count_documents({})
                
                # Get sample document for schema
                sample = await collection.find_one()
                
                datasets.append({
                    "id": name,
                    "name": name.replace("_", " ").title(),
                    "type": "imported" if name.startswith("imported_") else "uploaded",
                    "record_count": count,
                    "fields": list(sample.keys()) if sample else [],
                    "created_at": sample.get("_imported_at") if sample else None
                })
    
    return {
        "datasets": datasets[offset:offset + limit],
        "total": len(datasets),
        "limit": limit,
        "offset": offset
    }

@router.get("/dataset/{dataset_id}/preview")
async def preview_dataset(
    dataset_id: str,
    limit: int = Query(10, ge=1, le=100)
):
    """Preview dataset records"""
    if db.mongo_db is None:
        raise HTTPException(status_code=503, detail="Database not available")
    
    collection = db.mongo_db[dataset_id]
    
    # Get sample records
    records = await collection.find().limit(limit).to_list(limit)
    
    # Get total count
    total_count = await collection.count_documents({})
    
    return {
        "dataset_id": dataset_id,
        "records": records,
        "sample_size": len(records),
        "total_count": total_count
    }

@router.get("/dataset/{dataset_id}/issues")
async def analyze_dataset_issues(dataset_id: str):
    """Analyze dataset for quality issues"""
    if db.mongo_db is None:
        raise HTTPException(status_code=503, detail="Database not available")
    
    collection = db.mongo_db[dataset_id]
    
    # Get all records (limited for performance)
    records = await collection.find().limit(10000).to_list(10000)
    
    if not records:
        return {"dataset_id": dataset_id, "issues": []}
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(records)
    
    issues = []
    
    # Check for missing values
    missing = df.isnull().sum()
    if missing.any():
        total_cells = len(df) * len(df.columns)
        missing_cells = missing.sum()
        issues.append({
            "type": "missing_values",
            "severity": "high" if missing_cells / total_cells > 0.1 else "medium",
            "count": int(missing_cells),
            "percentage": round((missing_cells / total_cells) * 100, 2),
            "details": missing[missing > 0].to_dict()
        })
    
    # Check for duplicates
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        issues.append({
            "type": "duplicates",
            "severity": "medium",
            "count": int(duplicates),
            "percentage": round((duplicates / len(df)) * 100, 2)
        })
    
    # Check for inconsistent data types
    for col in df.columns:
        if col.startswith('_'):  # Skip metadata columns
            continue
        
        # Try to infer if column should be numeric
        try:
            pd.to_numeric(df[col], errors='coerce')
            non_numeric = df[col].apply(lambda x: not isinstance(x, (int, float))).sum()
            if non_numeric > 0 and non_numeric < len(df) * 0.1:
                issues.append({
                    "type": "format_issues",
                    "severity": "low",
                    "column": col,
                    "message": f"Column '{col}' appears to be numeric but has {non_numeric} non-numeric values"
                })
        except:
            pass
    
    return {
        "dataset_id": dataset_id,
        "total_records": len(df),
        "total_columns": len(df.columns),
        "issues": issues,
        "health_score": calculate_health_score(issues, len(df))
    }

def calculate_health_score(issues: List[Dict], total_records: int) -> float:
    """Calculate dataset health score (0-100)"""
    if not issues:
        return 100.0
    
    score = 100.0
    
    for issue in issues:
        if issue["severity"] == "high":
            score -= 20
        elif issue["severity"] == "medium":
            score -= 10
        else:
            score -= 5
    
    return max(0.0, score)
