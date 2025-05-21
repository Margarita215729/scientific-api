from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
from fastapi.middleware.cors import CORSMiddleware
import pathlib

# Import API modules
from api.astro_catalog_api import router as astro_router
from api.ads_api import router as ads_router
from api.dataset_api import router as dataset_router
from api.file_manager_api import router as file_router
from api.ml_models import router as ml_router

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

# Get the directory of the current file
current_dir = pathlib.Path(__file__).parent.parent
ui_dir = current_dir / "ui"

# Mount static files
app.mount("/static", StaticFiles(directory=str(ui_dir)), name="static")

# Set up Jinja2 templates
templates = Jinja2Templates(directory=str(ui_dir))

# Include routers
app.include_router(astro_router, prefix="/astro", tags=["Astronomical Catalogs"])
app.include_router(ads_router, prefix="/ads", tags=["ADS Literature"])
app.include_router(dataset_router, prefix="/datasets", tags=["Datasets"])
app.include_router(file_router, prefix="/files", tags=["File Management"])
app.include_router(ml_router, prefix="/ml", tags=["Machine Learning Models"])

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/ads")
async def ads_page(request: Request):
    return templates.TemplateResponse("ads.html", {"request": request})

@app.get("/api")
async def api_info():
    return {
        "message": "Scientific API",
        "version": "1.0.0",
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