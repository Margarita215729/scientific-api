from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import httpx
import os
import json
from typing import Dict, Any
import asyncio

app = FastAPI(title="Scientific API - Vercel Frontend")

# Azure backend URL
AZURE_BACKEND_URL = "http://scientific-api-full-1748121289.eastus.azurecontainer.io:8000"

# Добавляем монтирование статических файлов из директории ui
app.mount("/static", StaticFiles(directory="ui"), name="static")

# Health check endpoint
@app.get("/ping")
async def ping():
    return {"status": "ok", "service": "vercel-frontend"}

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "backend": AZURE_BACKEND_URL}

@app.get("/api")
async def root():
    return {
        "message": "Scientific API - Vercel Frontend",
        "backend": AZURE_BACKEND_URL,
        "endpoints": {
            "astro": "/api/astro/*",
            "ml": "/api/ml/*", 
            "ads": "/api/ads/*"
        }
    }

# Proxy function for Azure backend
async def proxy_to_azure(path: str, method: str, request: Request) -> Dict[str, Any]:
    """Proxy requests to Azure backend"""
    try:
        url = f"{AZURE_BACKEND_URL}/{path}"
        
        # Get query parameters
        query_params = dict(request.query_params)
        
        # Get request body if present
        body = None
        if method in ["POST", "PUT", "PATCH"]:
            try:
                body = await request.body()
                if body:
                    body = json.loads(body.decode())
            except:
                body = None
        
        # Make request to Azure backend
        async with httpx.AsyncClient(timeout=30.0) as client:
            if method == "GET":
                response = await client.get(url, params=query_params)
            elif method == "POST":
                response = await client.post(url, params=query_params, json=body)
            elif method == "PUT":
                response = await client.put(url, params=query_params, json=body)
            elif method == "DELETE":
                response = await client.delete(url, params=query_params)
            else:
                raise HTTPException(status_code=405, detail="Method not allowed")
        
        # Return response
        try:
            return response.json()
        except:
            return {"data": response.text, "status_code": response.status_code}
            
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Backend timeout")
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="Backend unavailable")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Proxy error: {str(e)}")

# Astro API proxy
@app.api_route("/api/astro/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_astro_to_azure(path: str, request: Request):
    return await proxy_to_azure(f"api/astro/{path}", request.method, request)

# ML API proxy  
@app.api_route("/api/ml/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_ml_to_azure(path: str, request: Request):
    return await proxy_to_azure(f"api/ml/{path}", request.method, request)

# ADS API proxy
@app.api_route("/api/ads/{path:path}", methods=["GET", "POST"])
async def proxy_ads_to_azure(path: str, request: Request):
    return await proxy_to_azure(f"api/ads/{path}", request.method, request)

# ---- NEW PROXY ROUTES WITHOUT /api PREFIX ----

@app.api_route("/astro/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_astro_no_api(path: str, request: Request):
    """Proxy /astro/* to Azure backend (maps to /api/astro/*)"""
    return await proxy_to_azure(f"api/astro/{path}", request.method, request)

@app.api_route("/ml/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_ml_no_api(path: str, request: Request):
    return await proxy_to_azure(f"api/ml/{path}", request.method, request)

# ADS auxiliary mappings
@app.api_route("/ads/search-by-catalog", methods=["GET"])
async def proxy_ads_catalog(request: Request):
    catalog = request.query_params.get("catalog", "")
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(f"{AZURE_BACKEND_URL}/api/ads/search-by-catalog", params={"catalog": catalog})
    try:
        return response.json()
    except:
        return {"data": response.text, "status_code": response.status_code}

# Generic fallback for /ads/* that are already correct
@app.api_route("/ads/{path:path}", methods=["GET", "POST"])
async def proxy_ads_no_api(path: str, request: Request):
    return await proxy_to_azure(f"api/ads/{path}", request.method, request)

@app.api_route("/ads/search-by-object", methods=["GET"])
async def proxy_ads_object(request: Request):
    object_name = request.query_params.get("object_name", "")
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(f"{AZURE_BACKEND_URL}/api/ads/search-by-object", params={"object_name": object_name})
    try:
        return response.json()
    except:
        return {"data": response.text, "status_code": response.status_code}

@app.api_route("/ads/search-by-coordinates", methods=["GET"])
async def proxy_ads_coordinates(request: Request):
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(f"{AZURE_BACKEND_URL}/api/ads/search-by-coordinates", params=request.query_params)
    try:
        return response.json()
    except:
        return {"data": response.text, "status_code": response.status_code}

@app.api_route("/ads/large-scale-structure", methods=["GET"])
async def proxy_ads_lss(request: Request):
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(f"{AZURE_BACKEND_URL}/api/ads/large-scale-structure", params=request.query_params)
    try:
        return response.json()
    except:
        return {"data": response.text, "status_code": response.status_code}

# Static file serving
@app.get("/")
async def serve_index():
    # Отдаём полноценный интерфейс, если он присутствует
    return FileResponse("ui/index.html")

@app.get("/ads")
async def serve_ads():
    return FileResponse("ui/ads.html")

@app.get("/static/{file_path:path}")
async def serve_static(file_path: str):
    try:
        return FileResponse(f"ui/{file_path}")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found")

@app.get("/favicon.ico")
async def favicon_ico():
    if os.path.exists("ui/favicon.ico"):
        return FileResponse("ui/favicon.ico")
    raise HTTPException(status_code=404, detail="Not found")

@app.get("/favicon.png")
async def favicon_png():
    if os.path.exists("ui/favicon.png"):
        return FileResponse("ui/favicon.png")
    raise HTTPException(status_code=404, detail="Not found") 