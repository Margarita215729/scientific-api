"""
Main FastAPI application for Vercel deployment.
Scientific API for astronomical data analysis.
"""

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import sys

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import API routers
try:
    from api.astro_catalog_api import router as astro_router
    from api.ads_api import router as ads_router
    from api.heavy_api import router as heavy_router
except ImportError as e:
    print(f"Warning: Could not import some API modules: {e}")
    astro_router = None
    ads_router = None
    heavy_router = None

app = FastAPI(
    title="Scientific API - Astronomical Data",
    description="API for accessing and analyzing astronomical catalogs and scientific literature",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
if astro_router:
    app.include_router(astro_router, prefix="/api/astro", tags=["Astronomical Catalogs"])
if ads_router:
    app.include_router(ads_router, prefix="/api/ads", tags=["Scientific Literature"])
if heavy_router:
    app.include_router(heavy_router, prefix="/api", tags=["Heavy Compute"])

# Health check endpoint
@app.get("/api/health")
async def health_check():
    """Health check endpoint for monitoring."""
    return {
        "status": "healthy",
        "service": "Scientific API",
        "version": "2.0.0",
        "environment": "production"
    }

# Root endpoint
@app.get("/api")
async def root():
    """Root API endpoint with service information."""
    return {
        "service": "Scientific API - Astronomical Data",
        "version": "2.0.0",
        "description": "Access to real astronomical catalogs and scientific literature",
        "status": "operational",
        "endpoints": {
            "health": "/api/health",
            "docs": "/api/docs",
            "astronomical_data": "/api/astro",
            "literature_search": "/api/ads"
        }
    }

# Static files and UI routes
@app.get("/")
async def serve_index():
    """Serve the main UI page."""
    try:
        with open("ui/index.html", "r", encoding="utf-8") as f:
            content = f.read()
        return HTMLResponse(content=content)
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Scientific API</h1><p>UI not found</p>")

@app.get("/ads")
async def serve_ads():
    """Serve the ADS search page."""
    try:
        with open("ui/ads.html", "r", encoding="utf-8") as f:
            content = f.read()
        return HTMLResponse(content=content)
    except FileNotFoundError:
        return HTMLResponse(content="<h1>ADS Search</h1><p>Page not found</p>")

@app.get("/static/{file_path:path}")
async def serve_static(file_path: str):
    """Serve static files."""
    static_file_path = f"ui/{file_path}"
    if os.path.exists(static_file_path):
        return FileResponse(static_file_path)
    raise HTTPException(status_code=404, detail="File not found")

# Vercel handler
def handler(request, context):
    """Vercel serverless function handler."""
    return app(request, context)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 