from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os

# Initialize FastAPI app
app = FastAPI(title="Scientific API for Azure")

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
        "message": "Scientific API is running on Azure Container Instances!", 
        "version": "1.0.0",
        "platform": "AMD64"
    }

@app.get("/api")
async def api_info():
    return {
        "message": "Scientific API",
        "version": "1.0.0",
        "environment": "Azure Container Instances",
        "endpoints": {
            "ping": "/ping - Check if API is running",
            "astro": "/astro/... - Astronomical data endpoints",
            "ads": "/ads/... - ADS Literature API"
        }
    }

# Import simple router that doesn't depend on pandas
try:
    from api.simple_router import router as simple_router
    app.include_router(simple_router, prefix="/astro", tags=["Simple Astronomical Data"])
    print("Successfully loaded simple_router")
except ImportError as e:
    print(f"Warning: Could not import simple_router: {e}")

# Try to import other routers
try:
    from api.ads_api import router as ads_router
    app.include_router(ads_router, prefix="/ads", tags=["ADS Literature"])
    print("Successfully loaded ads_router")
except ImportError as e:
    print(f"Warning: Could not import ads_router: {e}")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
