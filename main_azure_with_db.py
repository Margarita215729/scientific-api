from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
from contextlib import asynccontextmanager
import pandas as pd
import asyncio

# Import existing API modules
from api.astro_catalog_api import router as astro_router
from api.ads_api import router as ads_router
from api.ml_analysis_api import router as ml_router

# Import database
from database.config import db

async def load_csv_into_db(csv_path: str):
    """Bulk insert astronomical objects from merged CSV into database if the table is empty."""
    if not os.path.exists(csv_path):
        print(f"[startup] CSV not found: {csv_path}")
        return

    # Check current count
    res = await db.execute_query("SELECT COUNT(*) AS cnt FROM astronomical_objects")
    if res and res[0]["cnt"] > 0:
        print(f"[startup] astronomical_objects already filled ({res[0]['cnt']} rows)")
        return

    print("[startup] Loading astronomical_objects into database…")
    insert_sql = (
        "INSERT INTO astronomical_objects "
        "(RA, DEC, redshift, source, X, Y, Z) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)"
    )

    inserted = 0
    for chunk in pd.read_csv(csv_path, chunksize=1000):
        rows = [
            (
                float(r["RA"]),
                float(r["DEC"]),
                float(r["redshift"]),
                r["source"],
                float(r.get("X", 0)),
                float(r.get("Y", 0)),
                float(r.get("Z", 0)),
            )
            for _, r in chunk.iterrows()
        ]
        for row in rows:
            await db.execute_query(insert_sql, row)
        inserted += len(rows)
        if inserted % 10000 == 0:
            print(f"[startup] Inserted {inserted} rows…")

    print(f"[startup] Finished inserting {inserted} objects into DB")


async def ensure_data_ready():
    """Ensure astronomical data are processed and loaded into DB."""
    from api.heavy_api import load_preprocessed_data, process_astronomical_data, PROCESSED_DIR

    preprocess_info = load_preprocessed_data()
    if preprocess_info.get("status") != "processed":
        print("[startup] Preprocessed data not found → running heavy pipeline…")
        await process_astronomical_data("startup-task")
    else:
        print("[startup] Preprocessed data already present → skipping heavy pipeline")

    merged_csv = os.path.join(PROCESSED_DIR, "merged_catalog.csv")
    await load_csv_into_db(merged_csv)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    print("🚀 Starting Scientific API with Database…")
    await db.init_database()
    print("✅ Database initialized")
    
    # Ensure data are present
    try:
        await ensure_data_ready()
    except Exception as e:
        print(f"[startup] Data preparation error: {e}")
    
    yield
    
    # Shutdown
    await db.disconnect()
    print("🔌 Database disconnected")

app = FastAPI(
    title="Scientific API - Azure Backend with Database",
    description="Complete astronomical data analysis API with database integration",
    version="2.0.0",
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

@app.get("/")
async def serve_index():
    """Serve main page"""
    try:
        with open("static/index.html", "r", encoding="utf-8") as f:
            content = f.read()
        return HTMLResponse(content=content)
    except FileNotFoundError:
        return HTMLResponse(content="""
        <html>
            <head><title>Scientific API</title></head>
            <body>
                <h1>🔬 Scientific API - Azure Backend</h1>
                <p>Database-powered astronomical data analysis</p>
                <ul>
                    <li><a href="/docs">API Documentation</a></li>
                    <li><a href="/api/astro/status">Astro Status</a></li>
                    <li><a href="/api/database/status">Database Status</a></li>
                </ul>
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
    """Health check endpoint"""
    return {"status": "ok", "service": "azure-backend-with-db", "version": "2.0.0"}

@app.get("/api/health")
async def health_check():
    """Comprehensive health check"""
    try:
        # Test database connection
        stats = await db.get_statistics()
        db_status = "connected" if stats else "error"
    except Exception as e:
        db_status = f"error: {str(e)}"
    
    return {
        "status": "healthy",
        "service": "azure-backend-with-db",
        "database": db_status,
        "version": "2.0.0"
    }

@app.get("/api/database/status")
async def database_status():
    """Database status and statistics"""
    try:
        stats = await db.get_statistics()
        objects = await db.get_astronomical_objects(limit=5)
        return {
            "status": "connected",
            "statistics": stats,
            "sample_objects": len(objects),
            "database_type": db.db_type
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

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

@app.post("/api/database/cache")
async def cache_data(cache_key: str, data: dict, expires_hours: int = 24):
    """Cache data in database"""
    try:
        await db.cache_api_response(cache_key, data, expires_hours)
        return {"status": "cached", "key": cache_key, "expires_hours": expires_hours}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cache error: {str(e)}")

@app.get("/api/database/cache/{cache_key}")
async def get_cached_data(cache_key: str):
    """Get cached data from database"""
    try:
        data = await db.get_cached_response(cache_key)
        if data:
            return {"status": "hit", "data": data}
        else:
            return {"status": "miss", "data": None}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cache error: {str(e)}")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port) 