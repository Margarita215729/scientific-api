from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os

# Initialize FastAPI app
app = FastAPI(title="Scientific API for Azure - Full Version")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/ping")
async def ping():
    return {
        "status": "ok", 
        "message": "Scientific API is running on Azure Container Instances with full functionality!", 
        "version": "1.0.0",
        "platform": "AMD64"
    }

@app.get("/api")
async def api_info():
    return {
        "message": "Scientific API - Full Version",
        "version": "1.0.0",
        "environment": "Azure Container Instances",
        "endpoints": {
            "ping": "/ping - Check if API is running",
            "astro": "/astro/... - Astronomical data endpoints",
            "astro_full": "/astro/full/... - Full astronomical catalog endpoints",
            "ads": "/ads/... - ADS Literature API",
            "datasets": "/datasets/... - Dataset management",
            "files": "/files/... - File management",
            "ml": "/ml/... - Machine Learning models"
        }
    }

# Import routers - starting with simple ones that don't require heavy dependencies
try:
    from api.simple_router import router as simple_router
    app.include_router(simple_router, prefix="/astro", tags=["Simple Astronomical Data"])
    print("✅ Successfully loaded simple_router")
except ImportError as e:
    print(f"❌ Warning: Could not import simple_router: {e}")

try:
    from api.ads_api import router as ads_router
    app.include_router(ads_router, prefix="/ads", tags=["ADS Literature"])
    print("✅ Successfully loaded ads_router")
except ImportError as e:
    print(f"❌ Warning: Could not import ads_router: {e}")

# Import heavy computation routers
try:
    from api.astro_catalog_api import router as astro_router
    app.include_router(astro_router, prefix="/astro/full", tags=["Astronomical Catalogs"])
    print("✅ Successfully loaded full astro_catalog_api")
except ImportError as e:
    print(f"❌ Warning: Could not import astro_catalog_api: {e}")

try:
    from api.dataset_api import router as dataset_router
    app.include_router(dataset_router, prefix="/datasets", tags=["Datasets"])
    print("✅ Successfully loaded dataset_api")
except ImportError as e:
    print(f"❌ Warning: Could not import dataset_api: {e}")

try:
    from api.file_manager_api import router as file_router
    app.include_router(file_router, prefix="/files", tags=["File Management"])
    print("✅ Successfully loaded file_manager_api")
except ImportError as e:
    print(f"❌ Warning: Could not import file_manager_api: {e}")

try:
    from api.ml_models import router as ml_router
    app.include_router(ml_router, prefix="/ml", tags=["Machine Learning Models"])
    print("✅ Successfully loaded ml_models")
except ImportError as e:
    print(f"❌ Warning: Could not import ml_models: {e}")

# Import data analysis router
try:
    from api.data_analysis import router as data_analysis_router
    app.include_router(data_analysis_router, prefix="/analysis", tags=["Data Analysis"])
    print("✅ Successfully loaded data_analysis")
except ImportError as e:
    print(f"❌ Warning: Could not import data_analysis: {e}")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port) 