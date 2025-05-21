# api/index.py
# This file re-exports the FastAPI app from vercel_app.py for Vercel deployment
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import os
from fastapi.middleware.cors import CORSMiddleware
import logging

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Scientific API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup static files and get UI directory
from api.static_files import setup_static_files
ui_dir = setup_static_files(app)

# Set up Jinja2 templates
templates = Jinja2Templates(directory=str(ui_dir))

# Check dependencies
from api.dependencies import print_dependency_status
print_dependency_status()

# Import simple router that doesn't depend on pandas
from api.simple_router import router as simple_router
app.include_router(simple_router, prefix="/astro", tags=["Simple Astronomical Data"])

# Try to import routers, but don't fail if they're not available
try:
    from api.astro_catalog_api import router as astro_router
    app.include_router(astro_router, prefix="/astro/full", tags=["Astronomical Catalogs"])
    logger.info("Successfully loaded full astro_router")
except ImportError as e:
    logger.warning(f"Could not import astro_router: {e}")

try:
    from api.ads_api import router as ads_router
    app.include_router(ads_router, prefix="/ads", tags=["ADS Literature"])
    logger.info("Successfully loaded ads_router")
except ImportError as e:
    logger.warning(f"Could not import ads_router: {e}")

try:
    from api.dataset_api import router as dataset_router
    app.include_router(dataset_router, prefix="/datasets", tags=["Datasets"])
    logger.info("Successfully loaded dataset_router")
except ImportError as e:
    logger.warning(f"Could not import dataset_router: {e}")

try:
    from api.file_manager_api import router as file_router
    app.include_router(file_router, prefix="/files", tags=["File Management"])
    logger.info("Successfully loaded file_router")
except ImportError as e:
    logger.warning(f"Could not import file_router: {e}")

try:
    from api.ml_models import router as ml_router
    app.include_router(ml_router, prefix="/ml", tags=["Machine Learning Models"])
    logger.info("Successfully loaded ml_router")
except ImportError as e:
    logger.warning(f"Could not import ml_router: {e}")

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/ads")
async def ads_page(request: Request):
    return templates.TemplateResponse("ads.html", {"request": request})

@app.get("/api")
async def api_info():
    from api.dependencies import get_dependency_status
    
    return {
        "message": "Scientific API",
        "version": "1.0.0",
        "dependencies": get_dependency_status(),
        "endpoints": {
            "astro": "/astro/...",
            "ads": "/ads/...",
            "datasets": "/datasets/...",
            "files": "/files/...",
            "ml": "/ml/..."
        }
    }

@app.get("/ping")
async def ping():
    return {"status": "ok", "message": "API is up and running"} 