"""
ArXiv API integration for fetching scientific preprints.
Provides access to the latest research papers in various fields.
"""

import requests
import feedparser
import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import xml.etree.ElementTree as ET
import re

logger = logging.getLogger(__name__)

class ArXivAPI:
    """Client for ArXiv API."""
    
    def __init__(self):
        self.base_url = "http://export.arxiv.org/api/query"
        
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
        
        # ArXiv subject classifications
        self.subject_classes = {
            "astro-ph": "Astrophysics",
            "astro-ph.CO": "Cosmology and Nongalactic Astrophysics",
            "astro-ph.EP": "Earth and Planetary Astrophysics",
            "astro-ph.GA": "Astrophysics of Galaxies",
            "astro-ph.HE": "High Energy Astrophysical Phenomena",
            "astro-ph.IM": "Instrumentation and Methods for Astrophysics",
            "astro-ph.SR": "Solar and Stellar Astrophysics",
            "physics": "Physics",
            "physics.space-ph": "Space Physics",
            "gr-qc": "General Relativity and Quantum Cosmology",
            "hep-ph": "High Energy Physics - Phenomenology",
            "hep-th": "High Energy Physics - Theory",
            "math-ph": "Mathematical Physics",
            "cond-mat": "Condensed Matter",
            "cs": "Computer Science",
            "math": "Mathematics",
            "q-bio": "Quantitative Biology",
            "stat": "Statistics"
        }
    
    def _parse_authors(self, authors_text: str) -> List[str]:
        """Parse authors from ArXiv format."""
        if not authors_text:
            return []
        
        # Split by 'and' or commas, clean up
        authors = re.split(r',|\sand\s', authors_text)
        return [author.strip() for author in authors if author.strip()]
    
    def _extract_subjects(self, categories: str) -> List[str]:
        """Extract subject classifications from categories."""
        if not categories:
            return []
        
        subjects = []
        cats = categories.split()
        
        for cat in cats:
            if cat in self.subject_classes:
                subjects.append(self.subject_classes[cat])
            else:
                # Handle subcategories
                main_cat = cat.split('.')[0]
                if main_cat in self.subject_classes:
                    subjects.append(self.subject_classes[main_cat])
                else:
                    subjects.append(cat)  # Keep original if not found
        
        return list(set(subjects))  # Remove duplicates
    
    async def search_papers(self, query: str, max_results: int = 20, 
                          sort_by: str = "relevance", sort_order: str = "descending") -> Dict[str, Any]:
        """Search ArXiv for papers."""
        try:
            params = {
                "search_query": query,
                "start": 0,
                "max_results": min(max_results, 2000),  # ArXiv limit
                "sortBy": sort_by,
                "sortOrder": sort_order
            }
            
            response = self.session.get(self.base_url, params=params, timeout=30)
            
            if response.status_code == 200:
                feed = feedparser.parse(response.text)
                
                papers = []
                for entry in feed.entries:
                    # Extract ArXiv ID
                    arxiv_id = entry.get("id", "").split("/")[-1]
                    
                    # Parse authors
                    authors = []
                    if hasattr(entry, 'authors'):
                        authors = [author.name for author in entry.authors]
                    elif hasattr(entry, 'author'):
                        authors = [entry.author]
                    
                    # Extract publication date
                    published = entry.get("published", "")
                    pub_date = None
                    if published:
                        try:
                            pub_date = datetime.strptime(published[:10], "%Y-%m-%d").date().isoformat()
                        except:
                            pub_date = published[:10] if len(published) >= 10 else published
                    
                    # Extract categories/subjects
                    categories = entry.get("tags", [])
                    subjects = []
                    if categories:
                        for tag in categories:
                            if hasattr(tag, 'term'):
                                subjects.extend(self._extract_subjects(tag.term))
                    
                    # Get PDF link
                    pdf_url = None
                    if hasattr(entry, 'links'):
                        for link in entry.links:
                            if link.get('type') == 'application/pdf':
                                pdf_url = link.get('href')
                                break
                    
                    paper = {
                        "id": arxiv_id,
                        "arxiv_id": arxiv_id,
                        "title": entry.get("title", "").strip(),
                        "abstract": entry.get("summary", "").strip(),
                        "authors": authors,
                        "published_date": pub_date,
                        "updated_date": entry.get("updated", "")[:10] if entry.get("updated") else None,
                        "subjects": subjects,
                        "categories": [tag.term for tag in categories if hasattr(tag, 'term')],
                        "pdf_url": pdf_url,
                        "arxiv_url": entry.get("id", ""),
                        "doi": entry.get("arxiv_doi", ""),
                        "journal_ref": entry.get("arxiv_journal_ref", ""),
                        "comment": entry.get("arxiv_comment", ""),
                        "source": "ArXiv"
                    }
                    papers.append(paper)
                
                return {
                    "papers": papers,
                    "total_found": len(papers),
                    "query": query,
                    "status": "success"
                }
            
            else:
                logger.error(f"ArXiv API error: {response.status_code}")
                return {
                    "papers": [],
                    "total_found": 0,
                    "query": query,
                    "status": "error",
                    "error": f"HTTP {response.status_code}"
                }
                
        except Exception as e:
            logger.error(f"Error searching ArXiv: {e}", exc_info=True)
            # Return demo data for development
            return await self._generate_demo_data(query, max_results)
    
    async def get_paper_by_id(self, arxiv_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific paper by ArXiv ID."""
        try:
            # Clean ArXiv ID (remove version if present)
            clean_id = arxiv_id.split('v')[0] if 'v' in arxiv_id else arxiv_id
            
            result = await self.search_papers(f"id:{clean_id}", max_results=1)
            
            if result["papers"]:
                return result["papers"][0]
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error getting ArXiv paper {arxiv_id}: {e}")
            return None
    
    async def search_by_category(self, category: str, max_results: int = 20) -> Dict[str, Any]:
        """Search papers by ArXiv category."""
        try:
            query = f"cat:{category}"
            return await self.search_papers(query, max_results, sort_by="submittedDate")
            
        except Exception as e:
            logger.error(f"Error searching ArXiv category {category}: {e}")
            return {
                "papers": [],
                "total_found": 0,
                "query": query,
                "status": "error",
                "error": str(e)
            }
    
    async def search_astronomy_papers(self, query: str = "", max_results: int = 50) -> Dict[str, Any]:
        """Search specifically for astronomy and astrophysics papers."""
        try:
            # Astronomy categories
            astro_categories = [
                "astro-ph.CO", "astro-ph.EP", "astro-ph.GA", 
                "astro-ph.HE", "astro-ph.IM", "astro-ph.SR"
            ]
            
            # Construct category filter
            cat_filter = " OR ".join([f"cat:{cat}" for cat in astro_categories])
            
            if query:
                search_query = f"{query} AND ({cat_filter})"
            else:
                search_query = f"({cat_filter})"
            
            result = await self.search_papers(search_query, max_results, sort_by="submittedDate")
            result["search_type"] = "astronomy_focused"
            
            return result
            
        except Exception as e:
            logger.error(f"Error searching astronomy papers on ArXiv: {e}")
            return {
                "papers": [],
                "total_found": 0,
                "query": query,
                "status": "error",
                "error": str(e)
            }
    
    async def search_recent_papers(self, days_back: int = 7, category: str = "astro-ph") -> Dict[str, Any]:
        """Search for recent papers in a category."""
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            # Format dates for ArXiv (YYYYMMDD)
            start_str = start_date.strftime("%Y%m%d")
            end_str = end_date.strftime("%Y%m%d")
            
            # Search query with date range
            query = f"cat:{category} AND submittedDate:[{start_str} TO {end_str}]"
            
            result = await self.search_papers(query, max_results=100, sort_by="submittedDate")
            result["search_type"] = "recent_papers"
            result["date_range"] = {
                "start": start_date.date().isoformat(),
                "end": end_date.date().isoformat(),
                "days": days_back
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error searching recent ArXiv papers: {e}")
            return {
                "papers": [],
                "total_found": 0,
                "query": f"Recent {category} papers",
                "status": "error",
                "error": str(e)
            }
    
    async def search_by_author(self, author_name: str, max_results: int = 20) -> Dict[str, Any]:
        """Search papers by author name."""
        try:
            # Format author query
            query = f'au:"{author_name}"'
            
            result = await self.search_papers(query, max_results, sort_by="submittedDate")
            result["search_type"] = "author_search"
            result["author"] = author_name
            
            return result
            
        except Exception as e:
            logger.error(f"Error searching ArXiv by author {author_name}: {e}")
            return {
                "papers": [],
                "total_found": 0,
                "query": query,
                "status": "error",
                "error": str(e)
            }
    
    def get_categories(self) -> Dict[str, str]:
        """Get available ArXiv categories."""
        return self.subject_classes.copy()
    
    async def _generate_demo_data(self, query: str, max_results: int) -> Dict[str, Any]:
        """Generate demo data for development when real API is not available."""
        logger.info(f"Generating demo ArXiv data for query: {query}")
        
        import datetime
        import random
        
        # Sample paper titles and abstracts
        sample_papers = [
            {
                "title": f"Deep Learning Approaches to {query.title()} Classification and Analysis",
                "abstract": f"This paper presents novel deep learning methods for analyzing {query} data. We propose a new architecture that achieves state-of-the-art performance on benchmark datasets.",
                "authors": ["Smith, J.", "Johnson, A.", "Williams, B."],
                "subjects": ["Computer Science", "Astrophysics"]
            },
            {
                "title": f"Statistical Methods for {query.title()} Detection in Large-Scale Surveys",
                "abstract": f"We develop new statistical techniques for detecting {query} in astronomical survey data. Our method shows significant improvements over existing approaches.",
                "authors": ["Brown, C.", "Davis, M.", "Wilson, K."],
                "subjects": ["Statistics", "Astronomy"]
            },
            {
                "title": f"Machine Learning Applications in {query.title()} Research",
                "abstract": f"This review covers recent advances in applying machine learning to {query} research, highlighting key challenges and future directions.",
                "authors": ["Taylor, R.", "Anderson, L.", "Thompson, P."],
                "subjects": ["Machine Learning", "Physics"]
            },
            {
                "title": f"Observational Constraints on {query.title()} Formation Models",
                "abstract": f"Using data from multiple telescopes, we place new constraints on models of {query} formation and evolution in the early universe.",
                "authors": ["Garcia, M.", "Lee, S.", "Martinez, A."],
                "subjects": ["Astrophysics", "Cosmology"]
            },
            {
                "title": f"Numerical Simulations of {query.title()} Dynamics",
                "abstract": f"We present high-resolution numerical simulations of {query} dynamics using advanced computational methods on supercomputers.",
                "authors": ["Kumar, V.", "Zhang, L.", "Patel, N."],
                "subjects": ["Computational Physics", "Astronomy"]
            }
        ]
        
        papers = []
        num_papers = min(max_results, len(sample_papers))
        
        for i in range(num_papers):
            paper_data = sample_papers[i % len(sample_papers)]
            
            # Generate realistic dates
            base_date = datetime.datetime.now() - datetime.timedelta(days=random.randint(1, 365))
            pub_date = base_date.strftime("%Y-%m-%d")
            
            paper = {
                "id": f"2024.{random.randint(1000, 9999)}.{random.randint(10000, 99999)}",
                "arxiv_id": f"2024.{random.randint(1000, 9999)}.{random.randint(10000, 99999)}",
                "title": paper_data["title"],
                "abstract": paper_data["abstract"],
                "authors": paper_data["authors"],
                "published_date": pub_date,
                "updated_date": pub_date,
                "subjects": paper_data["subjects"],
                "categories": ["astro-ph.GA", "cs.LG"],
                "pdf_url": f"https://arxiv.org/pdf/2024.{random.randint(1000, 9999)}.{random.randint(10000, 99999)}.pdf",
                "arxiv_url": f"https://arxiv.org/abs/2024.{random.randint(1000, 9999)}.{random.randint(10000, 99999)}",
                "doi": "",
                "journal_ref": "",
                "comment": "Demo data for development",
                "source": "ArXiv (Demo)"
            }
            papers.append(paper)
        
        return {
            "papers": papers,
            "total_found": len(papers),
            "query": query,
            "status": "demo_data",
            "note": "Demo data - real ArXiv API not accessible"
        }
    
    async def download_paper_pdf(self, arxiv_id: str, output_path: str) -> bool:
        """Download PDF of a paper (for future implementation)."""
        try:
            # Clean ArXiv ID
            clean_id = arxiv_id.split('v')[0] if 'v' in arxiv_id else arxiv_id
            pdf_url = f"https://arxiv.org/pdf/{clean_id}.pdf"
            
            response = self.session.get(pdf_url, timeout=60, stream=True)
            
            if response.status_code == 200:
                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                logger.info(f"Downloaded PDF for {arxiv_id} to {output_path}")
                return True
            else:
                logger.error(f"Failed to download PDF for {arxiv_id}: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Error downloading PDF for {arxiv_id}: {e}")
            return False

# Global instance
arxiv_api = ArXivAPI()

# Convenience functions for backward compatibility
async def search_arxiv(query: str, max_results: int = 20) -> Dict[str, Any]:
    """Search ArXiv for papers."""
    return await arxiv_api.search_papers(query, max_results)

async def get_arxiv_paper(arxiv_id: str) -> Optional[Dict[str, Any]]:
    """Get a specific ArXiv paper by ID."""
    return await arxiv_api.get_paper_by_id(arxiv_id)

async def search_arxiv_astronomy(query: str = "", max_results: int = 50) -> Dict[str, Any]:
    """Search ArXiv for astronomy papers."""
    return await arxiv_api.search_astronomy_papers(query, max_results)

async def get_recent_arxiv_papers(days_back: int = 7, category: str = "astro-ph") -> Dict[str, Any]:
    """Get recent ArXiv papers."""
    return await arxiv_api.search_recent_papers(days_back, category)
