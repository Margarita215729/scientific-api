from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import logging
from contextlib import asynccontextmanager
import pandas as pd
import asyncio

# Import existing API modules
from api.astro_catalog_api import router as astro_router
from api.ads_api import router as ads_router
from api.ml_analysis_api import router as ml_router
from api.heavy_api import router as heavy_api_router

# Import database
from database.config import db

# Import data preprocessor
from utils.data_preprocessor import AstronomicalDataPreprocessor

logger = logging.getLogger(__name__)

# Define if heavy pipeline should be run on startup
RUN_HEAVY_PIPELINE_ON_START = os.getenv("HEAVY_PIPELINE_ON_START", "false").lower() == 'true'

async def load_data_into_db_if_empty():
    """Load initial astronomical data into database if table is empty."""
    try:
        logger.info("Checking if astronomical_objects table is empty...")
        # Check current count
        res = await db.execute_query("SELECT COUNT(*) AS cnt FROM astronomical_objects")
        if res and res[0]["cnt"] > 0:
            logger.info(f"[startup] astronomical_objects already filled ({res[0]['cnt']} rows)")
            return

        logger.info("No objects found in database. Attempting to run data preprocessing pipeline...")
        preprocessor = AstronomicalDataPreprocessor()
        # Run full data preprocessing pipeline
        preprocessing_results = await preprocessor.preprocess_all_catalogs()
        logger.info(f"Data preprocessing pipeline completed. Results: {preprocessing_results}")
        
        # Additional check after pipeline
        final_check_objects = await db.get_astronomical_objects(limit=1)
        if final_check_objects:
            logger.info("Data successfully loaded into database after pipeline run.")
        else:
            logger.warning("Data preprocessing pipeline ran, but database still appears empty. Check pipeline logs.")

    except Exception as e:
        logger.error(f"Error during initial data load/check: {e}", exc_info=True)
        # Don't stop the startup process due to this, but log the issue

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("🚀 Starting Scientific API with Database (main_azure_with_db.py)...")
    # Startup
    try:
        await db.connect()
        logger.info("✅ Database connected.")
        # Initialize database schema (create tables/containers if they don't exist)
        await db.init_database()
        logger.info("✅ Database schema initialized/verified.")
    except Exception as e_db_init:
        logger.critical(f"❌ CRITICAL: Failed to connect to or initialize database: {e_db_init}", exc_info=True)
        # Depending on policy, can either stop the application or continue with limited functionality
        # raise RuntimeError(f"Failed to initialize database: {e_db_init}") from e_db_init # Example of stopping

    # Run heavy data processing if flag is set
    if RUN_HEAVY_PIPELINE_ON_START:
        logger.info("HEAVY_PIPELINE_ON_START is true. Starting initial data loading/preprocessing...")
        # Run in background so as not to block API startup for long
        asyncio.create_task(load_data_into_db_if_empty())
    else:
        logger.info("HEAVY_PIPELINE_ON_START is false. Skipping automatic heavy data pipeline on startup.")
        # Can add quick check for data presence and warn if none exist
        # quick_check_objects = await db.get_astronomical_objects(limit=1)
        # if not quick_check_objects:
        #     logger.warning("Database appears to be empty and HEAVY_PIPELINE_ON_START is false. "
        #                    "Data might need to be loaded manually via API endpoint or by setting the env var.")

    yield
    
    # Shutdown
    logger.info("🔌 Shutting down Scientific API and disconnecting database...")
    if db.cosmos_client or db.sql_connection: # Check if connection was established
        await db.disconnect()
        logger.info("🔌 Database disconnected.")
    else:
        logger.info("🔌 Database was not connected, no disconnect needed.")

app = FastAPI(
    title="Scientific API - Azure Backend with Database",
    description="Complete astronomical data analysis API with database integration and heavy processing capabilities.",
    version="2.1.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# Include API routers
app.include_router(astro_router, prefix="/api/astro", tags=["Astronomical Catalog"])
app.include_router(ads_router, prefix="/api/ads", tags=["ADS Search"])
app.include_router(ml_router, prefix="/api/ml", tags=["ML Analysis"])
app.include_router(heavy_api_router, prefix="/api/heavy", tags=["Heavy Data Processing"])

@app.get("/")
async def serve_backend_root():
    """Serve main page for the backend (e.g., status or link to docs)"""
    return HTMLResponse(content="""
    <html>
        <head><title>Scientific API - Azure Backend</title></head>
        <body>
            <h1>🔬 Scientific API - Azure Backend</h1>
            <p>This is the main backend service with database integration.</p>
            <p>Key functionalities are available via API endpoints.</p>
            <ul>
                <li><a href="/docs">API Documentation (Swagger UI)</a></li>
                <li><a href="/redoc">API Documentation (ReDoc)</a></li>
                <li><a href="/api/health">Health Check</a></li>
                <li><a href="/api/database/status">Database Status</a></li>
            </ul>
            <p>Frontend UI is typically served via Vercel or another static host.</p>
        </body>
    </html>
    """)

@app.get("/ads")
async def serve_ads():
    """Serve ADS search page"""
    try:
        with open("static/ads.html", "r", encoding="utf-8") as f:
            content = f.read()
        return HTMLResponse(content=content)
    except FileNotFoundError:
        return HTMLResponse(content="""
        <html>
            <head><title>ADS Search</title></head>
            <body>
                <h1>📚 ADS Search</h1>
                <p>Search astronomical literature</p>
                <a href="/">← Back to main</a>
            </body>
        </html>
        """)

@app.get("/ping")
async def ping():
    """Simple health check endpoint."""
    return {"status": "ok", "service": "scientific-api-azure-backend", "version": app.version}

@app.get("/api/health", tags=["System"])
async def health_check():
    """Comprehensive health check including database connectivity."""
    db_status = "unknown"
    db_type_info = "N/A"
    if db.cosmos_client or db.sql_connection: # Check if connection was successful
        try:
            # Try a simple query to the DB, for example, getting statistics
            await db.get_statistics() # This method should already be working
            db_status = "connected"
            db_type_info = db.db_type
        except Exception as e_db_check:
            db_status = f"error_during_check: {str(e_db_check)[:100]}..."
            logger.warning(f"Health check: DB connection seems to exist, but test query failed: {e_db_check}", exc_info=True)
    else:
        db_status = "not_connected_or_initialized_yet"
    
    return {
        "status": "healthy" if db_status == "connected" else "degraded",
        "service_name": app.title,
        "version": app.version,
        "database_status": db_status,
        "database_type": db_type_info,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }

@app.get("/api/database/status", tags=["Database"])
async def database_status():
    """Get current database status and basic statistics."""
    if not (db.cosmos_client or db.sql_connection):
        raise HTTPException(status_code=503, detail="Database not connected or initialized.")
    try:
        stats = await db.get_statistics()
        # Get a few objects for example, to ensure tables/containers are accessible
        objects_sample = await db.get_astronomical_objects(limit=3)
        return {
            "status": "connected",
            "database_type": db.db_type,
            "database_name": db.cosmos_database_name if db.db_type == "cosmosdb" else db.db_url,
            "statistics_preview": {k: v for i, (k, v) in enumerate(stats.items()) if i < 5}, # First 5 metrics
            "sample_objects_count": len(objects_sample),
        }
    except Exception as e:
        logger.error(f"Error getting database status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Database status check error: {str(e)}")

@app.get("/api/database/objects")
async def get_database_objects(limit: int = 100, object_type: str = None):
    """Get astronomical objects from database"""
    try:
        objects = await db.get_astronomical_objects(limit=limit, object_type=object_type)
        return {
            "objects": objects,
            "count": len(objects),
            "limit": limit,
            "filter": object_type
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.post("/api/database/cache", tags=["Cache"])
async def cache_data_endpoint(cache_key: str, data: dict, expires_hours: int = 24):
    """Endpoint to cache data in the database."""
    if not (db.cosmos_client or db.sql_connection):
        raise HTTPException(status_code=503, detail="Database not connected for caching.")
    try:
        await db.cache_api_response(cache_key, data, expires_hours)
        return {"status": "cached", "key": cache_key, "expires_hours": expires_hours}
    except Exception as e:
        logger.error(f"Error caching data for key '{cache_key}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Cache error: {str(e)}")

@app.get("/api/database/cache/{cache_key}", tags=["Cache"])
async def get_cached_data_endpoint(cache_key: str):
    """Endpoint to retrieve cached data from the database."""
    if not (db.cosmos_client or db.sql_connection):
        raise HTTPException(status_code=503, detail="Database not connected for cache retrieval.")
    try:
        data = await db.get_cached_response(cache_key)
        if data is not None:
            return {"status": "hit", "key": cache_key, "data": data}
        else:
            # Return 404 if key not found in cache, this is more standard for REST
            raise HTTPException(status_code=404, detail=f"Cache miss for key: {cache_key}")
    except HTTPException: # Catch 404 to not log as server error
        raise
    except Exception as e:
        logger.error(f"Error retrieving cached data for key '{cache_key}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Cache retrieval error: {str(e)}")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    # Settings for uvicorn, workers can be moved to configuration (e.g., for Gunicorn)
    uvicorn.run(
        "main_azure_with_db:app", 
        host="0.0.0.0", 
        port=port, 
        reload=os.getenv("DEBUG_RELOAD", "false").lower() == 'true', # Enable reload only if DEBUG_RELOAD=true
        workers=int(os.getenv("WEB_CONCURRENCY", 1)) # Number of workers, 1 for SQLite
    ) 