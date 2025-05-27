"""
Azure Cosmos DB configuration for caching and data storage.
"""

# pyright: reportMissingImports=false

import os
import logging
from typing import Dict, Any, Optional, List
import json
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Azure Cosmos DB configuration
COSMOS_DB_ENDPOINT = os.getenv("COSMOS_DB_ENDPOINT", "")
COSMOS_DB_KEY = os.getenv("COSMOS_DB_KEY", "")
COSMOS_DB_DATABASE = os.getenv("COSMOS_DB_DATABASE", "scientific-data")
COSMOS_DB_CONTAINER = os.getenv("COSMOS_DB_CONTAINER", "cache")

# Cache settings
CACHE_TTL_HOURS = int(os.getenv("CACHE_TTL_HOURS", "24"))
ENABLE_COSMOS_CACHE = bool(COSMOS_DB_ENDPOINT and COSMOS_DB_KEY)

class CosmosDBCache:
    """Azure Cosmos DB cache implementation."""
    
    def __init__(self):
        self.client = None
        self.database = None
        self.container = None
        self.enabled = ENABLE_COSMOS_CACHE
        
        if self.enabled:
            try:
                from azure.cosmos import CosmosClient
                self.client = CosmosClient(COSMOS_DB_ENDPOINT, COSMOS_DB_KEY)
                self.database = self.client.get_database_client(COSMOS_DB_DATABASE)
                self.container = self.database.get_container_client(COSMOS_DB_CONTAINER)
                logger.info("Cosmos DB cache initialized successfully")
            except ImportError:
                logger.warning("azure-cosmos library not available, cache disabled")
                self.enabled = False
            except Exception as e:
                logger.error(f"Failed to initialize Cosmos DB: {e}")
                self.enabled = False
        else:
            logger.info("Cosmos DB cache disabled (missing configuration)")
    
    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get cached data by key."""
        if not self.enabled:
            return None
        
        try:
            item = self.container.read_item(item=key, partition_key=key)
            
            # Check if cache is expired
            expires_at = datetime.fromisoformat(item.get("expires_at", ""))
            if datetime.utcnow() > expires_at:
                await self.delete(key)
                return None
            
            return item.get("data")
        except Exception as e:
            logger.debug(f"Cache miss for key {key}: {e}")
            return None
    
    async def set(self, key: str, data: Dict[str, Any], ttl_hours: Optional[int] = None) -> bool:
        """Set cached data with TTL."""
        if not self.enabled:
            return False
        
        try:
            ttl = ttl_hours or CACHE_TTL_HOURS
            expires_at = datetime.utcnow() + timedelta(hours=ttl)
            
            item = {
                "id": key,
                "data": data,
                "created_at": datetime.utcnow().isoformat(),
                "expires_at": expires_at.isoformat(),
                "ttl": ttl * 3600  # Cosmos DB TTL in seconds
            }
            
            self.container.upsert_item(item)
            logger.debug(f"Cached data for key {key}")
            return True
        except Exception as e:
            logger.error(f"Failed to cache data for key {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete cached data."""
        if not self.enabled:
            return False
        
        try:
            self.container.delete_item(item=key, partition_key=key)
            logger.debug(f"Deleted cache for key {key}")
            return True
        except Exception as e:
            logger.debug(f"Failed to delete cache for key {key}: {e}")
            return False
    
    async def clear_expired(self) -> int:
        """Clear expired cache entries."""
        if not self.enabled:
            return 0
        
        try:
            # Cosmos DB automatically handles TTL, but we can query for manual cleanup
            query = "SELECT c.id FROM c WHERE c.expires_at < @now"
            parameters = [{"name": "@now", "value": datetime.utcnow().isoformat()}]
            
            expired_items = list(self.container.query_items(
                query=query,
                parameters=parameters,
                enable_cross_partition_query=True
            ))
            
            count = 0
            for item in expired_items:
                if await self.delete(item["id"]):
                    count += 1
            
            logger.info(f"Cleared {count} expired cache entries")
            return count
        except Exception as e:
            logger.error(f"Failed to clear expired cache: {e}")
            return 0

# Global cache instance
cosmos_cache = CosmosDBCache()

# Cache decorators and utilities
def cache_key(*args, **kwargs) -> str:
    """Generate cache key from arguments."""
    key_parts = [str(arg) for arg in args]
    key_parts.extend([f"{k}={v}" for k, v in sorted(kwargs.items())])
    return ":".join(key_parts)

async def cached_api_call(key: str, api_func, *args, ttl_hours: int = 24, **kwargs):
    """Execute API call with caching."""
    # Try to get from cache first
    cached_result = await cosmos_cache.get(key)
    if cached_result is not None:
        logger.debug(f"Cache hit for {key}")
        return cached_result
    
    # Execute API call
    logger.debug(f"Cache miss for {key}, executing API call")
    result = await api_func(*args, **kwargs)
    
    # Cache the result
    await cosmos_cache.set(key, result, ttl_hours)
    return result

# Specific cache functions for astronomical data
async def cache_catalog_data(catalog_name: str, filters: Dict, data: List[Dict]) -> bool:
    """Cache catalog data with filters."""
    key = cache_key("catalog", catalog_name, **filters)
    return await cosmos_cache.set(key, {"data": data, "count": len(data)})

async def get_cached_catalog_data(catalog_name: str, filters: Dict) -> Optional[List[Dict]]:
    """Get cached catalog data."""
    key = cache_key("catalog", catalog_name, **filters)
    cached = await cosmos_cache.get(key)
    return cached.get("data") if cached else None

async def cache_ads_search(query: str, search_type: str, results: List[Dict]) -> bool:
    """Cache ADS search results."""
    key = cache_key("ads", search_type, query)
    return await cosmos_cache.set(key, {"results": results, "count": len(results)}, ttl_hours=6)

async def get_cached_ads_search(query: str, search_type: str) -> Optional[List[Dict]]:
    """Get cached ADS search results."""
    key = cache_key("ads", search_type, query)
    cached = await cosmos_cache.get(key)
    return cached.get("results") if cached else None

async def cache_statistics(stats: Dict[str, Any]) -> bool:
    """Cache statistical data."""
    key = cache_key("statistics", "global")
    return await cosmos_cache.set(key, stats, ttl_hours=12)

async def get_cached_statistics() -> Optional[Dict[str, Any]]:
    """Get cached statistics."""
    key = cache_key("statistics", "global")
    return await cosmos_cache.get(key) 