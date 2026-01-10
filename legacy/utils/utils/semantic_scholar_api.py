"""
Semantic Scholar API integration for fetching scientific papers.
Provides access to academic publications with citation data and metadata.
"""

import requests
import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
import time

logger = logging.getLogger(__name__)

class SemanticScholarAPI:
    """Client for Semantic Scholar API."""
    
    def __init__(self):
        self.base_url = "https://api.semanticscholar.org/graph/v1"
        
        # Use SSL configuration utility
        try:
            from utils.ssl_config import configure_ssl_for_requests
            self.session = configure_ssl_for_requests()
        except ImportError:
            # Fallback to basic session
            self.session = requests.Session()
            self.session.verify = False
            try:
                import urllib3
                urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            except ImportError:
                pass
        
        self.session.headers.update({
            "User-Agent": "Scientific-API/1.0 (https://scientific-api.com)"
        })
        
        self.rate_limit_delay = 1.0  # 1 second between requests
        self.last_request_time = 0
    
    def _wait_for_rate_limit(self):
        """Implement rate limiting."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - time_since_last)
        self.last_request_time = time.time()
    
    async def search_papers(self, query: str, max_results: int = 20, fields: List[str] = None) -> Dict[str, Any]:
        """Search for papers using Semantic Scholar API."""
        try:
            if fields is None:
                fields = [
                    "paperId", "title", "abstract", "authors", "year", "venue",
                    "citationCount", "referenceCount", "doi", "url", "publicationDate",
                    "publicationTypes", "fieldsOfStudy", "s2FieldsOfStudy"
                ]
            
            # Rate limiting
            self._wait_for_rate_limit()
            
            url = f"{self.base_url}/paper/search"
            params = {
                "query": query,
                "limit": min(max_results, 100),  # API limit is 100
                "fields": ",".join(fields)
            }
            
            response = self.session.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                papers = data.get("data", [])
                
                # Process papers
                processed_papers = []
                for paper in papers:
                    processed_paper = {
                        "id": paper.get("paperId"),
                        "title": paper.get("title"),
                        "abstract": paper.get("abstract"),
                        "authors": [author.get("name", "Unknown") for author in paper.get("authors", [])],
                        "year": paper.get("year"),
                        "venue": paper.get("venue"),
                        "citation_count": paper.get("citationCount", 0),
                        "reference_count": paper.get("referenceCount", 0),
                        "doi": paper.get("doi"),
                        "url": paper.get("url"),
                        "publication_date": paper.get("publicationDate"),
                        "publication_types": paper.get("publicationTypes", []),
                        "fields_of_study": paper.get("fieldsOfStudy", []),
                        "s2_fields": [field.get("category") for field in paper.get("s2FieldsOfStudy", [])],
                        "source": "Semantic Scholar"
                    }
                    processed_papers.append(processed_paper)
                
                return {
                    "papers": processed_papers,
                    "total_found": data.get("total", len(processed_papers)),
                    "query": query,
                    "status": "success"
                }
            
            elif response.status_code == 429:  # Rate limited
                logger.warning("Rate limited by Semantic Scholar API")
                await asyncio.sleep(5)  # Wait longer for rate limit
                return await self.search_papers(query, max_results, fields)  # Retry
            
            else:
                logger.error(f"Semantic Scholar API error: {response.status_code} - {response.text}")
                return {
                    "papers": [],
                    "total_found": 0,
                    "query": query,
                    "status": "error",
                    "error": f"API error: {response.status_code}"
                }
                
        except Exception as e:
            logger.error(f"Error searching Semantic Scholar: {e}", exc_info=True)
            return {
                "papers": [],
                "total_found": 0,
                "query": query,
                "status": "error",
                "error": str(e)
            }
    
    async def get_paper_details(self, paper_id: str, fields: List[str] = None) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific paper."""
        try:
            if fields is None:
                fields = [
                    "paperId", "title", "abstract", "authors", "year", "venue",
                    "citationCount", "referenceCount", "doi", "url", "publicationDate",
                    "publicationTypes", "fieldsOfStudy", "s2FieldsOfStudy", "citations",
                    "references", "embedding"
                ]
            
            self._wait_for_rate_limit()
            
            url = f"{self.base_url}/paper/{paper_id}"
            params = {"fields": ",".join(fields)}
            
            response = self.session.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                paper = response.json()
                
                processed_paper = {
                    "id": paper.get("paperId"),
                    "title": paper.get("title"),
                    "abstract": paper.get("abstract"),
                    "authors": [
                        {
                            "name": author.get("name", "Unknown"),
                            "author_id": author.get("authorId"),
                            "affiliations": author.get("affiliations", [])
                        }
                        for author in paper.get("authors", [])
                    ],
                    "year": paper.get("year"),
                    "venue": paper.get("venue"),
                    "citation_count": paper.get("citationCount", 0),
                    "reference_count": paper.get("referenceCount", 0),
                    "doi": paper.get("doi"),
                    "url": paper.get("url"),
                    "publication_date": paper.get("publicationDate"),
                    "publication_types": paper.get("publicationTypes", []),
                    "fields_of_study": paper.get("fieldsOfStudy", []),
                    "s2_fields": [field.get("category") for field in paper.get("s2FieldsOfStudy", [])],
                    "citations": paper.get("citations", []),
                    "references": paper.get("references", []),
                    "embedding": paper.get("embedding"),
                    "source": "Semantic Scholar"
                }
                
                return processed_paper
            
            else:
                logger.error(f"Error getting paper {paper_id}: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting paper details {paper_id}: {e}")
            return None
    
    async def get_citations(self, paper_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get papers that cite the given paper."""
        try:
            self._wait_for_rate_limit()
            
            url = f"{self.base_url}/paper/{paper_id}/citations"
            params = {
                "limit": min(limit, 1000),
                "fields": "paperId,title,authors,year,venue,citationCount"
            }
            
            response = self.session.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                citations = []
                
                for citation in data.get("data", []):
                    citing_paper = citation.get("citingPaper", {})
                    citations.append({
                        "id": citing_paper.get("paperId"),
                        "title": citing_paper.get("title"),
                        "authors": [author.get("name", "Unknown") for author in citing_paper.get("authors", [])],
                        "year": citing_paper.get("year"),
                        "venue": citing_paper.get("venue"),
                        "citation_count": citing_paper.get("citationCount", 0)
                    })
                
                return citations
            
            else:
                logger.error(f"Error getting citations for {paper_id}: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Error getting citations for {paper_id}: {e}")
            return []
    
    async def get_references(self, paper_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get papers referenced by the given paper."""
        try:
            self._wait_for_rate_limit()
            
            url = f"{self.base_url}/paper/{paper_id}/references"
            params = {
                "limit": min(limit, 1000),
                "fields": "paperId,title,authors,year,venue,citationCount"
            }
            
            response = self.session.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                references = []
                
                for reference in data.get("data", []):
                    cited_paper = reference.get("citedPaper", {})
                    references.append({
                        "id": cited_paper.get("paperId"),
                        "title": cited_paper.get("title"),
                        "authors": [author.get("name", "Unknown") for author in cited_paper.get("authors", [])],
                        "year": cited_paper.get("year"),
                        "venue": cited_paper.get("venue"),
                        "citation_count": cited_paper.get("citationCount", 0)
                    })
                
                return references
            
            else:
                logger.error(f"Error getting references for {paper_id}: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Error getting references for {paper_id}: {e}")
            return []
    
    async def search_by_field(self, field: str, query: str = "", max_results: int = 20) -> Dict[str, Any]:
        """Search papers by specific field of study."""
        try:
            # Construct query with field filter
            search_query = f"fieldsOfStudy:{field}"
            if query:
                search_query = f"{query} AND {search_query}"
            
            return await self.search_papers(search_query, max_results)
            
        except Exception as e:
            logger.error(f"Error searching by field {field}: {e}")
            return {
                "papers": [],
                "total_found": 0,
                "query": search_query,
                "status": "error",
                "error": str(e)
            }
    
    async def search_astronomy_papers(self, query: str = "", max_results: int = 50) -> Dict[str, Any]:
        """Search specifically for astronomy and astrophysics papers."""
        try:
            # Astronomy-related fields
            astronomy_fields = [
                "Physics", "Astronomy", "Astrophysics", "Cosmology",
                "Planetary science", "Space science"
            ]
            
            # Construct query for astronomy papers
            field_filters = " OR ".join([f"fieldsOfStudy:{field}" for field in astronomy_fields])
            
            if query:
                search_query = f"{query} AND ({field_filters})"
            else:
                search_query = f"({field_filters})"
            
            result = await self.search_papers(search_query, max_results)
            result["search_type"] = "astronomy_focused"
            
            return result
            
        except Exception as e:
            logger.error(f"Error searching astronomy papers: {e}")
            return {
                "papers": [],
                "total_found": 0,
                "query": query,
                "status": "error",
                "error": str(e)
            }

# Global instance
semantic_scholar_api = SemanticScholarAPI()

# Convenience functions for backward compatibility
async def search_semantic_scholar(query: str, max_results: int = 20) -> Dict[str, Any]:
    """Search Semantic Scholar for papers."""
    return await semantic_scholar_api.search_papers(query, max_results)

async def get_paper_citations(paper_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    """Get citations for a paper."""
    return await semantic_scholar_api.get_citations(paper_id, limit)

async def get_paper_references(paper_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    """Get references for a paper."""
    return await semantic_scholar_api.get_references(paper_id, limit)

async def search_astronomy_papers(query: str = "", max_results: int = 50) -> Dict[str, Any]:
    """Search for astronomy papers."""
    return await semantic_scholar_api.search_astronomy_papers(query, max_results)
