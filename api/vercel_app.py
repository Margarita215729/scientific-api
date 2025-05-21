from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
from fastapi.middleware.cors import CORSMiddleware
import pathlib

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

# Import simple router that doesn't depend on pandas
from api.simple_router import router as simple_router
app.include_router(simple_router, prefix="/astro", tags=["Simple Astronomical Data"])

# Try to import routers, but don't fail if they're not available
try:
    from api.astro_catalog_api import router as astro_router
    app.include_router(astro_router, prefix="/astro/full", tags=["Astronomical Catalogs"])
    print("Successfully loaded full astro_router")
except ImportError as e:
    print(f"Warning: Could not import astro_router: {e}")

try:
    from api.ads_api import router as ads_router
    app.include_router(ads_router, prefix="/ads", tags=["ADS Literature"])
    print("Successfully loaded ads_router")
except ImportError as e:
    print(f"Warning: Could not import ads_router: {e}")

try:
    from api.dataset_api import router as dataset_router
    app.include_router(dataset_router, prefix="/datasets", tags=["Datasets"])
    print("Successfully loaded dataset_router")
except ImportError as e:
    print(f"Warning: Could not import dataset_router: {e}")

try:
    from api.file_manager_api import router as file_router
    app.include_router(file_router, prefix="/files", tags=["File Management"])
    print("Successfully loaded file_router")
except ImportError as e:
    print(f"Warning: Could not import file_router: {e}")

try:
    from api.ml_models import router as ml_router
    app.include_router(ml_router, prefix="/ml", tags=["Machine Learning Models"])
    print("Successfully loaded ml_router")
except ImportError as e:
    print(f"Warning: Could not import ml_router: {e}")

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