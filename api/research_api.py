"""
Research API integration module.
Provides unified access to multiple scientific paper databases including
Semantic Scholar, ArXiv, and ADS for comprehensive research data collection.
"""

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks, Depends
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
import asyncio
import uuid

# Security imports
try:
    from api.security import (
        rate_limit_check, get_current_user, require_authentication,
        InputValidator, create_secure_response
    )
    SECURITY_AVAILABLE = True
except ImportError:
    SECURITY_AVAILABLE = False
    # Fallback functions
    async def rate_limit_check(request=None):
        return "no_limit"
    async def get_current_user():
        return {"user_type": "anonymous", "permissions": ["read"]}
    async def require_authentication():
        return {"user_type": "anonymous", "permissions": ["read"]}
    InputValidator = None
    def create_secure_response(data, status_code=200):
        return JSONResponse(content=data, status_code=status_code)

# Import our API clients
try:
    from utils.semantic_scholar_api import semantic_scholar_api
    SEMANTIC_SCHOLAR_AVAILABLE = True
except ImportError:
    semantic_scholar_api = None
    SEMANTIC_SCHOLAR_AVAILABLE = False

try:
    from utils.arxiv_api import arxiv_api
    ARXIV_AVAILABLE = True
except ImportError:
    arxiv_api = None
    ARXIV_AVAILABLE = False

try:
    from utils.ads_astronomy_real import ads_client
    ADS_AVAILABLE = True
except ImportError:
    ads_client = None
    ADS_AVAILABLE = False

# Database integration for caching
try:
    from database.config import db
    DB_AVAILABLE = True
except ImportError:
    db = None
    DB_AVAILABLE = False

logger = logging.getLogger(__name__)
router = APIRouter()

# Global storage for background tasks
research_tasks_status: Dict[str, Dict[str, Any]] = {}

@router.get("/status", tags=["Research API"])
async def get_research_api_status():
    """Get status of all research API integrations."""
    return {
        "status": "operational",
        "integrations": {
            "semantic_scholar": {
                "available": SEMANTIC_SCHOLAR_AVAILABLE,
                "description": "Academic paper search with citation data"
            },
            "arxiv": {
                "available": ARXIV_AVAILABLE,
                "description": "Preprint repository access"
            },
            "ads": {
                "available": ADS_AVAILABLE,
                "description": "Astrophysics Data System integration"
            }
        },
        "database_caching": DB_AVAILABLE,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }

@router.get("/search", tags=["Research Search"])
async def unified_research_search(
    query: str = Query(..., description="Search query"),
    sources: List[str] = Query(["semantic_scholar", "arxiv", "ads"], description="Data sources to search"),
    max_results_per_source: int = Query(20, ge=1, le=100, description="Maximum results per source"),
    cache_results: bool = Query(True, description="Cache results in database"),
    client_id: str = Depends(rate_limit_check),
    current_user: dict = Depends(get_current_user)
):
    """
    Unified search across multiple research databases.
    Searches Semantic Scholar, ArXiv, and ADS simultaneously.
    """
    try:
        # Validate and sanitize input
        if SECURITY_AVAILABLE and InputValidator:
            query = InputValidator.validate_query_string(query)
            max_results_per_source = InputValidator.validate_limit(max_results_per_source, 100)
        
        # Fix sources parameter - convert single string to list if needed
        if isinstance(sources, str):
            sources = [s.strip() for s in sources.split(",")]
        
        logger.info(f"Research search - query: {query}, sources: {sources}")
        
        results = {
            "query": query,
            "sources_searched": [],
            "total_papers": 0,
            "papers_by_source": {},
            "search_timestamp": datetime.utcnow().isoformat() + "Z",
            "status": "success"
        }
        
        search_tasks = []
        
        # Semantic Scholar search
        if "semantic_scholar" in sources and SEMANTIC_SCHOLAR_AVAILABLE:
            search_tasks.append(("semantic_scholar", semantic_scholar_api.search_papers(query, max_results_per_source)))
            results["sources_searched"].append("semantic_scholar")
        
        # ArXiv search
        if "arxiv" in sources and ARXIV_AVAILABLE:
            search_tasks.append(("arxiv", arxiv_api.search_papers(query, max_results_per_source)))
            results["sources_searched"].append("arxiv")
        
        # ADS search
        if "ads" in sources and ADS_AVAILABLE:
            search_tasks.append(("ads", ads_client.search_publications(query, max_results_per_source)))
            results["sources_searched"].append("ads")
        
        if not search_tasks:
            # Provide detailed error information
            api_status = {
                "semantic_scholar": SEMANTIC_SCHOLAR_AVAILABLE,
                "arxiv": ARXIV_AVAILABLE, 
                "ads": ADS_AVAILABLE,
                "requested_sources": sources
            }
            raise HTTPException(
                status_code=503, 
                detail=f"No research APIs are available. Status: {api_status}"
            )
        
        # Execute searches concurrently
        search_results = await asyncio.gather(*[task[1] for task in search_tasks], return_exceptions=True)
        
        # Process results
        for i, (source_name, search_result) in enumerate(zip([task[0] for task in search_tasks], search_results)):
            if isinstance(search_result, Exception):
                logger.error(f"Error searching {source_name}: {search_result}")
                results["papers_by_source"][source_name] = {
                    "status": "error",
                    "error": str(search_result),
                    "papers": []
                }
            else:
                if source_name == "ads":
                    # ADS returns different format
                    papers = search_result.get("publications", [])
                    results["papers_by_source"][source_name] = {
                        "status": "success",
                        "count": len(papers),
                        "papers": papers
                    }
                else:
                    # Semantic Scholar and ArXiv use similar format
                    papers = search_result.get("papers", [])
                    results["papers_by_source"][source_name] = {
                        "status": "success",
                        "count": len(papers),
                        "papers": papers
                    }
                
                results["total_papers"] += len(papers)
        
        # Cache results if requested and database is available
        if cache_results and DB_AVAILABLE:
            try:
                cache_key = f"research_search_{hash(query)}_{hash(str(sources))}"
                await db.cache_api_response(cache_key, results, expires_in_hours=24)
            except Exception as e:
                logger.warning(f"Failed to cache search results: {e}")
        
        return create_secure_response(results)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in unified research search: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@router.get("/search/astronomy", tags=["Research Search"])
async def search_astronomy_papers(
    query: str = Query("", description="Search query (optional for general astronomy papers)"),
    max_results: int = Query(50, ge=1, le=200, description="Maximum total results"),
    include_preprints: bool = Query(True, description="Include ArXiv preprints"),
    include_published: bool = Query(True, description="Include published papers")
):
    """
    Search specifically for astronomy and astrophysics papers across all sources.
    """
    try:
        sources = []
        if include_published:
            sources.extend(["semantic_scholar", "ads"])
        if include_preprints:
            sources.append("arxiv")
        
        results_per_source = max_results // len(sources) if sources else max_results
        
        search_tasks = []
        
        # Semantic Scholar astronomy search
        if "semantic_scholar" in sources and SEMANTIC_SCHOLAR_AVAILABLE:
            search_tasks.append(("semantic_scholar", semantic_scholar_api.search_astronomy_papers(query, results_per_source)))
        
        # ArXiv astronomy search
        if "arxiv" in sources and ARXIV_AVAILABLE:
            search_tasks.append(("arxiv", arxiv_api.search_astronomy_papers(query, results_per_source)))
        
        # ADS search (already astronomy-focused)
        if "ads" in sources and ADS_AVAILABLE:
            ads_query = query if query else "astronomy astrophysics"
            search_tasks.append(("ads", ads_client.search_publications(ads_query, results_per_source)))
        
        if not search_tasks:
            raise HTTPException(status_code=503, detail="No astronomy research APIs are available")
        
        # Execute searches concurrently
        search_results = await asyncio.gather(*[task[1] for task in search_tasks], return_exceptions=True)
        
        # Process and combine results
        all_papers = []
        source_stats = {}
        
        for i, (source_name, search_result) in enumerate(zip([task[0] for task in search_tasks], search_results)):
            if isinstance(search_result, Exception):
                logger.error(f"Error searching astronomy papers in {source_name}: {search_result}")
                source_stats[source_name] = {"status": "error", "count": 0, "error": str(search_result)}
            else:
                if source_name == "ads":
                    papers = search_result.get("publications", [])
                else:
                    papers = search_result.get("papers", [])
                
                # Add source information to each paper
                for paper in papers:
                    paper["data_source"] = source_name
                
                all_papers.extend(papers)
                source_stats[source_name] = {"status": "success", "count": len(papers)}
        
        # Sort by relevance/citation count if available
        def sort_key(paper):
            citation_count = paper.get("citation_count", 0) or paper.get("citationCount", 0)
            return citation_count if citation_count else 0
        
        all_papers.sort(key=sort_key, reverse=True)
        
        # Limit total results
        if len(all_papers) > max_results:
            all_papers = all_papers[:max_results]
        
        return {
            "query": query or "astronomy astrophysics papers",
            "total_papers": len(all_papers),
            "papers": all_papers,
            "source_statistics": source_stats,
            "search_timestamp": datetime.utcnow().isoformat() + "Z",
            "status": "success"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error searching astronomy papers: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Astronomy search failed: {str(e)}")

@router.get("/paper/{paper_id}/details", tags=["Research Details"])
async def get_paper_details(
    paper_id: str,
    source: str = Query(..., description="Source database (semantic_scholar, arxiv, ads)")
):
    """
    Get detailed information about a specific paper from a source database.
    """
    try:
        if source == "semantic_scholar" and SEMANTIC_SCHOLAR_AVAILABLE:
            paper = await semantic_scholar_api.get_paper_details(paper_id)
            if paper:
                return {"status": "success", "source": source, "paper": paper}
            else:
                raise HTTPException(status_code=404, detail="Paper not found in Semantic Scholar")
        
        elif source == "arxiv" and ARXIV_AVAILABLE:
            paper = await arxiv_api.get_paper_by_id(paper_id)
            if paper:
                return {"status": "success", "source": source, "paper": paper}
            else:
                raise HTTPException(status_code=404, detail="Paper not found in ArXiv")
        
        elif source == "ads" and ADS_AVAILABLE:
            # For ADS, we need to search by bibcode
            result = await ads_client.search_publications(f"bibcode:{paper_id}", max_results=1)
            papers = result.get("publications", [])
            if papers:
                return {"status": "success", "source": source, "paper": papers[0]}
            else:
                raise HTTPException(status_code=404, detail="Paper not found in ADS")
        
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported source: {source} or API not available")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting paper details {paper_id} from {source}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get paper details: {str(e)}")

@router.get("/paper/{paper_id}/citations", tags=["Research Details"])
async def get_paper_citations(
    paper_id: str,
    source: str = Query(..., description="Source database (semantic_scholar, ads)"),
    limit: int = Query(50, ge=1, le=200, description="Maximum number of citations")
):
    """
    Get papers that cite the specified paper.
    """
    try:
        if source == "semantic_scholar" and SEMANTIC_SCHOLAR_AVAILABLE:
            citations = await semantic_scholar_api.get_citations(paper_id, limit)
            return {
                "status": "success",
                "source": source,
                "paper_id": paper_id,
                "citation_count": len(citations),
                "citations": citations
            }
        
        elif source == "ads" and ADS_AVAILABLE:
            from utils.ads_astronomy_real import get_citations_for_paper
            citations = await get_citations_for_paper(paper_id)
            return {
                "status": "success",
                "source": source,
                "paper_id": paper_id,
                "citation_count": len(citations),
                "citations": citations
            }
        
        else:
            raise HTTPException(status_code=400, detail=f"Citations not supported for source: {source}")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting citations for {paper_id} from {source}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get citations: {str(e)}")

@router.post("/dataset/create", tags=["Research Dataset"])
async def create_research_dataset(
    background_tasks: BackgroundTasks,
    queries: List[str],
    sources: List[str] = ["semantic_scholar", "arxiv", "ads"],
    max_papers_per_query: int = 100,
    dataset_name: Optional[str] = None,
    current_user: dict = Depends(require_authentication),
    client_id: str = Depends(rate_limit_check)
):
    """
    Create a research dataset by collecting papers from multiple queries and sources.
    This is a background task that can take several minutes to complete.
    """
    try:
        # Generate task ID
        task_id = str(uuid.uuid4())
        
        if not dataset_name:
            dataset_name = f"research_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Initialize task status
        research_tasks_status[task_id] = {
            "status": "started",
            "message": "Initializing research dataset creation...",
            "dataset_name": dataset_name,
            "queries": queries,
            "sources": sources,
            "progress": 0,
            "total_papers": 0,
            "started_at": datetime.utcnow().isoformat()
        }
        
        # Start background task
        background_tasks.add_task(create_dataset_task, task_id, queries, sources, max_papers_per_query, dataset_name)
        
        return {
            "task_id": task_id,
            "status": "started",
            "message": "Research dataset creation started",
            "dataset_name": dataset_name,
            "estimated_duration": "5-15 minutes"
        }
        
    except Exception as e:
        logger.error(f"Error starting dataset creation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to start dataset creation: {str(e)}")

async def create_dataset_task(task_id: str, queries: List[str], sources: List[str], 
                            max_papers_per_query: int, dataset_name: str):
    """Background task for creating research dataset."""
    try:
        research_tasks_status[task_id]["status"] = "collecting"
        research_tasks_status[task_id]["message"] = "Collecting papers from research databases..."
        
        all_papers = []
        total_queries = len(queries)
        
        for i, query in enumerate(queries):
            research_tasks_status[task_id]["message"] = f"Processing query {i+1}/{total_queries}: {query}"
            research_tasks_status[task_id]["progress"] = int((i / total_queries) * 80)  # 80% for collection
            
            # Search each source for this query
            for source in sources:
                try:
                    if source == "semantic_scholar" and SEMANTIC_SCHOLAR_AVAILABLE:
                        result = await semantic_scholar_api.search_papers(query, max_papers_per_query)
                        papers = result.get("papers", [])
                    elif source == "arxiv" and ARXIV_AVAILABLE:
                        result = await arxiv_api.search_papers(query, max_papers_per_query)
                        papers = result.get("papers", [])
                    elif source == "ads" and ADS_AVAILABLE:
                        result = await ads_client.search_publications(query, max_papers_per_query)
                        papers = result.get("publications", [])
                    else:
                        continue
                    
                    # Add metadata to papers
                    for paper in papers:
                        paper["source_query"] = query
                        paper["data_source"] = source
                        paper["collected_at"] = datetime.utcnow().isoformat() + "Z"
                    
                    all_papers.extend(papers)
                    
                    # Small delay to respect rate limits
                    await asyncio.sleep(1)
                    
                except Exception as e:
                    logger.error(f"Error collecting from {source} for query '{query}': {e}")
        
        research_tasks_status[task_id]["status"] = "processing"
        research_tasks_status[task_id]["message"] = "Processing and deduplicating papers..."
        research_tasks_status[task_id]["progress"] = 85
        
        # Remove duplicates based on title similarity
        unique_papers = []
        seen_titles = set()
        
        for paper in all_papers:
            title = paper.get("title", "").lower().strip()
            if title and title not in seen_titles:
                seen_titles.add(title)
                unique_papers.append(paper)
        
        research_tasks_status[task_id]["total_papers"] = len(unique_papers)
        
        # Save to database if available
        if DB_AVAILABLE:
            research_tasks_status[task_id]["message"] = "Saving dataset to database..."
            research_tasks_status[task_id]["progress"] = 90
            
            try:
                # Store dataset metadata
                dataset_info = {
                    "dataset_name": dataset_name,
                    "queries": queries,
                    "sources": sources,
                    "total_papers": len(unique_papers),
                    "created_at": datetime.utcnow().isoformat() + "Z",
                    "task_id": task_id
                }
                
                # Cache the dataset
                cache_key = f"research_dataset_{task_id}"
                dataset_data = {
                    "info": dataset_info,
                    "papers": unique_papers
                }
                await db.cache_api_response(cache_key, dataset_data, expires_in_hours=24*7)  # 1 week
                
            except Exception as e:
                logger.error(f"Error saving dataset to database: {e}")
        
        research_tasks_status[task_id]["status"] = "completed"
        research_tasks_status[task_id]["message"] = "Research dataset created successfully"
        research_tasks_status[task_id]["progress"] = 100
        research_tasks_status[task_id]["completed_at"] = datetime.utcnow().isoformat()
        
        logger.info(f"Research dataset creation task {task_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Error in dataset creation task {task_id}: {e}", exc_info=True)
        research_tasks_status[task_id]["status"] = "failed"
        research_tasks_status[task_id]["message"] = f"Dataset creation failed: {str(e)}"
        research_tasks_status[task_id]["error"] = str(e)
        research_tasks_status[task_id]["completed_at"] = datetime.utcnow().isoformat()

@router.get("/dataset/task/{task_id}", tags=["Research Dataset"])
async def get_dataset_task_status(task_id: str):
    """Get the status of a dataset creation task."""
    if task_id not in research_tasks_status:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return research_tasks_status[task_id]

@router.get("/recent", tags=["Research Search"])
async def get_recent_papers(
    days_back: int = Query(7, ge=1, le=30, description="Days to look back"),
    sources: List[str] = Query(["arxiv"], description="Sources for recent papers"),
    category: str = Query("astro-ph", description="Category for ArXiv papers"),
    limit: int = Query(50, ge=1, le=200, description="Maximum number of papers")
):
    """
    Get recent papers from research databases.
    """
    try:
        recent_papers = []
        
        if "arxiv" in sources and ARXIV_AVAILABLE:
            arxiv_result = await arxiv_api.search_recent_papers(days_back, category)
            papers = arxiv_result.get("papers", [])
            for paper in papers:
                paper["data_source"] = "arxiv"
            recent_papers.extend(papers)
        
        # Sort by publication date (newest first)
        def sort_key(paper):
            pub_date = paper.get("published_date") or paper.get("publication_date", "")
            if pub_date:
                try:
                    return datetime.fromisoformat(pub_date.replace("Z", "+00:00"))
                except:
                    return datetime.min
            return datetime.min
        
        recent_papers.sort(key=sort_key, reverse=True)
        
        # Limit results
        if len(recent_papers) > limit:
            recent_papers = recent_papers[:limit]
        
        return {
            "status": "success",
            "days_back": days_back,
            "category": category,
            "sources": sources,
            "total_papers": len(recent_papers),
            "papers": recent_papers,
            "search_timestamp": datetime.utcnow().isoformat() + "Z"
        }
        
    except Exception as e:
        logger.error(f"Error getting recent papers: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get recent papers: {str(e)}")

@router.get("/categories", tags=["Research Info"])
async def get_available_categories():
    """Get available categories from all research sources."""
    categories = {
        "arxiv": arxiv_api.get_categories() if ARXIV_AVAILABLE else {},
        "semantic_scholar_fields": [
            "Computer Science", "Medicine", "Biology", "Physics", "Chemistry",
            "Mathematics", "Engineering", "Materials Science", "Psychology",
            "Economics", "Sociology", "Political Science", "Philosophy",
            "Art", "History", "Geography", "Geology", "Environmental Science",
            "Agricultural and Food Sciences", "Business"
        ] if SEMANTIC_SCHOLAR_AVAILABLE else [],
        "ads_databases": [
            "astronomy", "physics", "general"
        ] if ADS_AVAILABLE else []
    }
    
    return {
        "status": "success",
        "categories": categories,
        "available_sources": {
            "semantic_scholar": SEMANTIC_SCHOLAR_AVAILABLE,
            "arxiv": ARXIV_AVAILABLE,
            "ads": ADS_AVAILABLE
        }
    }
