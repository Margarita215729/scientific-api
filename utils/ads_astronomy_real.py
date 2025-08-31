"""
Enhanced NASA ADS (Astrophysics Data System) integration module.
Uses real ADSABS_TOKEN for searching astronomical publications.
"""

import os
import requests
import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import json

# Configure logging
logger = logging.getLogger(__name__)

# ADS API configuration
ADS_API_BASE = "https://api.adsabs.harvard.edu/v1"
ADS_TOKEN = os.getenv("ADSABS_TOKEN")

class ADSClient:
    """Client for NASA ADS API."""
    
    def __init__(self):
        if not ADS_TOKEN:
            logger.warning("ADSABS_TOKEN not found in environment variables")
            self.token = None
        else:
            self.token = ADS_TOKEN
            logger.info("ADS client initialized with token")
    
    def _get_headers(self) -> Dict[str, str]:
        """Get headers for ADS API requests."""
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Scientific-API/1.0"
        }
        
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        
        return headers
    
    async def search_publications(self, query: str, max_results: int = 20, 
                                 sort: str = "citation_count desc") -> Dict[str, Any]:
        """Search ADS for publications."""
        if not self.token:
            return await self._fallback_search(query, max_results)
        
        try:
            url = f"{ADS_API_BASE}/search/query"
            
            params = {
                "q": query,
                "fl": "bibcode,title,author,year,citation_count,doi,abstract,keyword,database",
                "rows": max_results,
                "sort": sort
            }
            
            headers = self._get_headers()
            
            # Make async request using executor
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, lambda: requests.get(url, params=params, headers=headers, timeout=30, verify=False)
            )
            
            if response.status_code == 200:
                data = response.json()
                return self._process_search_results(data, query)
            else:
                logger.error(f"ADS API error: {response.status_code} - {response.text}")
                return await self._fallback_search(query, max_results)
        
        except Exception as e:
            logger.error(f"Error searching ADS: {e}")
            return await self._fallback_search(query, max_results)
    
    async def search_by_coordinates(self, ra: float, dec: float, radius: float = 0.1,
                                   max_results: int = 20) -> Dict[str, Any]:
        """Search ADS for publications related to specific coordinates."""
        # Convert coordinates to proper format for ADS
        coord_query = f"object:\"RA {ra:.6f} DEC {dec:.6f}\""
        
        # Also search for nearby objects
        region_query = f"(ra:[{ra-radius:.6f} TO {ra+radius:.6f}] AND dec:[{dec-radius:.6f} TO {dec+radius:.6f}])"
        
        combined_query = f"({coord_query} OR {region_query}) AND database:astronomy"
        
        result = await self.search_publications(combined_query, max_results)
        result["search_params"] = {
            "ra": ra,
            "dec": dec,
            "radius": radius,
            "query_type": "coordinates"
        }
        
        return result
    
    async def search_by_object(self, object_name: str, max_results: int = 20) -> Dict[str, Any]:
        """Search ADS for publications about a specific astronomical object."""
        # Clean object name for better search
        clean_name = object_name.replace(" ", "+")
        
        # Search for object name in title, abstract, and keywords
        object_query = f"(title:\"{object_name}\" OR abstract:\"{object_name}\" OR keyword:\"{object_name}\" OR object:\"{object_name}\") AND database:astronomy"
        
        result = await self.search_publications(object_query, max_results)
        result["search_params"] = {
            "object_name": object_name,
            "query_type": "object"
        }
        
        return result
    
    async def search_by_catalog(self, catalog_name: str, max_results: int = 50) -> Dict[str, Any]:
        """Search ADS for publications related to a specific catalog."""
        # Search for catalog in various fields
        catalog_query = f"(title:\"{catalog_name}\" OR abstract:\"{catalog_name}\" OR keyword:\"{catalog_name}\") AND database:astronomy"
        
        result = await self.search_publications(catalog_query, max_results, sort="date desc")
        
        # Extract keyword statistics
        if "publications" in result:
            keywords = []
            for pub in result["publications"]:
                if "keyword" in pub and pub["keyword"]:
                    keywords.extend(pub["keyword"])
            
            # Count keyword occurrences
            keyword_counts = {}
            for keyword in keywords:
                if keyword:
                    keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
            
            # Get top keywords
            top_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:20]
            result["keyword_stats"] = dict(top_keywords)
        
        result["search_params"] = {
            "catalog": catalog_name,
            "query_type": "catalog"
        }
        
        return result
    
    async def search_large_scale_structure(self, start_year: int = 2010,
                                          additional_keywords: Optional[str] = None,
                                          max_results: int = 50) -> Dict[str, Any]:
        """Search ADS for papers on large-scale structure."""
        # Build query for large-scale structure topics
        lss_terms = [
            "large scale structure",
            "cosmic web",
            "galaxy clustering",
            "filaments",
            "voids",
            "superclusters",
            "dark matter halos",
            "redshift survey"
        ]
        
        # Create OR query for LSS terms
        lss_query = " OR ".join([f'"{term}"' for term in lss_terms])
        
        # Add year constraint
        year_query = f"year:[{start_year} TO 2024]"
        
        # Combine queries
        full_query = f"({lss_query}) AND {year_query} AND database:astronomy"
        
        # Add additional keywords if provided
        if additional_keywords:
            additional_query = " OR ".join([f'"{kw.strip()}"' for kw in additional_keywords.split(",")])
            full_query = f"({full_query}) AND ({additional_query})"
        
        result = await self.search_publications(full_query, max_results, sort="citation_count desc")
        
        # Extract year statistics
        if "publications" in result:
            year_counts = {}
            for pub in result["publications"]:
                year = pub.get("year")
                if year:
                    year_counts[year] = year_counts.get(year, 0) + 1
            
            result["year_stats"] = dict(sorted(year_counts.items()))
        
        result["search_params"] = {
            "start_year": start_year,
            "additional_keywords": additional_keywords,
            "query_type": "large_scale_structure"
        }
        
        return result
    
    def _process_search_results(self, data: Dict, original_query: str) -> Dict[str, Any]:
        """Process raw ADS search results."""
        try:
            if "response" not in data:
                return {"publications": [], "total_found": 0, "error": "Invalid response format"}
            
            response = data["response"]
            docs = response.get("docs", [])
            
            publications = []
            for doc in docs:
                pub = {
                    "title": doc.get("title", ["Unknown Title"]),
                    "author": doc.get("author", ["Unknown Author"]),
                    "year": doc.get("year", "Unknown"),
                    "citation_count": doc.get("citation_count", 0),
                    "bibcode": doc.get("bibcode", ""),
                    "doi": doc.get("doi", []),
                    "abstract": doc.get("abstract", ""),
                    "keyword": doc.get("keyword", []),
                    "database": doc.get("database", [])
                }
                
                # Clean up fields
                if isinstance(pub["title"], list) and pub["title"]:
                    pub["title"] = pub["title"][0]
                
                if isinstance(pub["doi"], list) and pub["doi"]:
                    pub["doi"] = pub["doi"][0]
                
                publications.append(pub)
            
            return {
                "publications": publications,
                "total_found": response.get("numFound", len(publications)),
                "query": original_query,
                "status": "success"
            }
        
        except Exception as e:
            logger.error(f"Error processing ADS results: {e}")
            return {"publications": [], "total_found": 0, "error": str(e)}
    
    async def _fallback_search(self, query: str, max_results: int) -> Dict[str, Any]:
        """Fallback search using alternative sources when ADS token is not available."""
        logger.warning(f"ADS token not available, using fallback search for query: {query}")
        
        try:
            # Try to use ArXiv as fallback
            import feedparser
            arxiv_url = "http://export.arxiv.org/api/query"
            params = {
                "search_query": query,
                "start": 0,
                "max_results": min(max_results, 20)
            }
            
            import requests
            response = requests.get(arxiv_url, params=params, timeout=10)
            if response.status_code == 200:
                feed = feedparser.parse(response.text)
                publications = []
                
                for i, entry in enumerate(feed.entries):
                    if i >= max_results:
                        break
                    
                    # Extract authors
                    authors = []
                    if hasattr(entry, 'authors'):
                        authors = [author.name for author in entry.authors]
                    elif hasattr(entry, 'author'):
                        authors = [entry.author]
                    
                    # Extract year from published date
                    year = "Unknown"
                    if hasattr(entry, 'published'):
                        try:
                            year = entry.published[:4]
                        except:
                            year = "2023"
                    
                    pub = {
                        "title": entry.get("title", "").strip(),
                        "author": authors,
                        "year": year,
                        "citation_count": 0,  # ArXiv doesn't provide citation counts
                        "bibcode": entry.get("id", "").split("/")[-1] if entry.get("id") else f"arXiv:{i+1}",
                        "doi": [],
                        "abstract": entry.get("summary", "").strip(),
                        "keyword": ["astronomy", "astrophysics", "preprint"],
                        "database": ["arXiv"]
                    }
                    publications.append(pub)
                
                return {
                    "publications": publications,
                    "total_found": len(publications),
                    "query": query,
                    "status": "fallback_arxiv",
                    "note": "Results from ArXiv (ADS token not available)"
                }
        
        except Exception as e:
            logger.error(f"Fallback search failed: {e}")
        
        # If all else fails, return empty results
        return {
            "publications": [],
            "total_found": 0,
            "query": query,
            "status": "no_results",
            "note": "No results available - ADS token required for full functionality"
        }

# Global ADS client instance
ads_client = ADSClient()

# Async wrapper functions for backward compatibility
async def search_ads_advanced(search_params: Dict) -> Dict[str, Any]:
    """Advanced ADS search with multiple options."""
    search_type = search_params.get("search_type", "general")
    
    if search_type == "coordinates":
        return await ads_client.search_by_coordinates(
            ra=search_params["ra"],
            dec=search_params["dec"],
            radius=search_params.get("radius", 0.1),
            max_results=search_params.get("max_results", 20)
        )
    
    elif search_type == "object":
        return await ads_client.search_by_object(
            object_name=search_params["query"],
            max_results=search_params.get("max_results", 20)
        )
    
    elif search_type == "catalog":
        return await ads_client.search_by_catalog(
            catalog_name=search_params["query"],
            max_results=search_params.get("max_results", 50)
        )
    
    elif search_type == "large_scale_structure":
        return await ads_client.search_large_scale_structure(
            start_year=search_params.get("start_year", 2010),
            additional_keywords=search_params.get("additional_keywords"),
            max_results=search_params.get("max_results", 50)
        )
    
    else:
        # General search
        return await ads_client.search_publications(
            query=search_params["query"],
            max_results=search_params.get("max_results", 20)
        )

# Wrapper functions for backward compatibility
async def search_by_coordinates(ra: float, dec: float, radius: float = 0.1, max_results: int = 20) -> List[Dict]:
    """Search ADS for publications related to specific coordinates."""
    result = await ads_client.search_by_coordinates(ra, dec, radius, max_results)
    return result.get("publications", [])

async def search_by_object(object_name: str, max_results: int = 20) -> List[Dict]:
    """Search ADS for publications about a specific astronomical object."""
    result = await ads_client.search_by_object(object_name, max_results)
    return result.get("publications", [])

async def search_by_catalog(catalog_name: str, facet: bool = True, max_results: int = 50) -> Dict[str, Any]:
    """Search ADS for publications related to a specific catalog."""
    result = await ads_client.search_by_catalog(catalog_name, max_results)
    return result

async def get_citations_for_paper(bibcode: str) -> List[Dict]:
    """Get citations for a specific paper."""
    query = f"citations(bibcode:{bibcode})"
    result = await ads_client.search_publications(query, max_results=100)
    return result.get("publications", [])

async def get_references_for_paper(bibcode: str) -> List[Dict]:
    """Get references for a specific paper."""
    query = f"references(bibcode:{bibcode})"
    result = await ads_client.search_publications(query, max_results=100)
    return result.get("publications", [])

async def search_large_scale_structure(additional_keywords: Optional[List[str]] = None, 
                                      start_year: int = 2010) -> Dict[str, Any]:
    """Search for publications on large-scale structure."""
    keywords_str = ",".join(additional_keywords) if additional_keywords else None
    result = await ads_client.search_large_scale_structure(start_year, keywords_str)
    return result

def get_bibtex(bibcodes: List[str]) -> str:
    """Get BibTeX format for given bibcodes using ADS API."""
    try:
        if not ADS_TOKEN:
            logger.warning("ADS token not available for BibTeX export")
            # Return a basic format with available information
            bibtex_entries = []
            for bibcode in bibcodes:
                entry = f"""@ARTICLE{{{bibcode.replace("&", "").replace(".", "").replace(" ", "")},
       bibcode = {{{bibcode}}},
        title = {{[Title not available - ADS token required]}},
       author = {{[Author not available - ADS token required]}},
      journal = {{[Journal not available - ADS token required]}},
         year = {{[Year not available - ADS token required]}},
       adsurl = {{https://ui.adsabs.harvard.edu/abs/{bibcode}}},
      adsnote = {{Provided by the SAO/NASA Astrophysics Data System}}
}}"""
                bibtex_entries.append(entry)
            return "\n\n".join(bibtex_entries)
        
        # Use ADS API to get BibTeX
        import requests
        url = f"{ADS_API_BASE}/export/bibtex"
        headers = {
            "Authorization": f"Bearer {ADS_TOKEN}",
            "Content-Type": "application/json"
        }
        
        payload = {"bibcode": bibcodes}
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            return data.get("export", "")
        else:
            logger.error(f"ADS BibTeX API error: {response.status_code}")
            raise Exception(f"ADS API error: {response.status_code}")
            
    except Exception as e:
        logger.error(f"Error getting BibTeX: {e}")
        # Fallback to basic format
        bibtex_entries = []
        for bibcode in bibcodes:
            entry = f"""@ARTICLE{{{bibcode.replace("&", "").replace(".", "").replace(" ", "")},
       bibcode = {{{bibcode}}},
        title = {{[Error retrieving title: {str(e)}]}},
       author = {{[Error retrieving author]}},
      journal = {{[Error retrieving journal]}},
         year = {{[Error retrieving year]}},
       adsurl = {{https://ui.adsabs.harvard.edu/abs/{bibcode}}},
      adsnote = {{Provided by the SAO/NASA Astrophysics Data System}}
}}"""
            bibtex_entries.append(entry)
        return "\n\n".join(bibtex_entries)

# Import numpy for mock data generation
try:
    import numpy as np
except ImportError:
    # Fallback if numpy not available
    class MockNumpy:
        @staticmethod
        def random():
            import random
            class MockRandom:
                @staticmethod
                def randint(a, b):
                    return random.randint(a, b)
            return MockRandom()
    np = MockNumpy() 