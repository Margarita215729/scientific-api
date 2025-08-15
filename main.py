from fastapi import FastAPI, Request, HTTPException, Form, Depends, status
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import httpx
import os
import json
from typing import Dict, Any, Optional
import asyncio
import secrets
import hashlib

app = FastAPI(title="Scientific API - Vercel Frontend")

# Azure backend URL
AZURE_BACKEND_URL = "https://scientific-api-e3a7a5dph6b3axa3.canadacentral-01.azurewebsites.net"

# Simple session management
SESSIONS = {}
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD_HASH = hashlib.sha256("admin123".encode()).hexdigest()  # Simple demo password

def generate_session_token():
    return secrets.token_urlsafe(32)

def get_current_user(request: Request) -> Optional[str]:
    session_token = request.cookies.get("session_token")
    if session_token and session_token in SESSIONS:
        return SESSIONS[session_token]["username"]
    return None

def require_admin(request: Request):
    user = get_current_user(request)
    if user != ADMIN_USERNAME:
        raise HTTPException(status_code=401, detail="Authentication required")

# Добавляем монтирование статических файлов из директории ui
app.mount("/static", StaticFiles(directory="ui"), name="static")

# Health check endpoint
@app.get("/ping")
async def ping():
    return {"status": "ok", "service": "vercel-frontend"}

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "backend": AZURE_BACKEND_URL}

# Authentication endpoints
@app.get("/login")
async def login_page():
    return FileResponse("ui/login.html")

@app.post("/login")
async def login(username: str = Form(...), password: str = Form(...)):
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    
    if username == ADMIN_USERNAME and password_hash == ADMIN_PASSWORD_HASH:
        session_token = generate_session_token()
        SESSIONS[session_token] = {"username": username}
        
        response = RedirectResponse(url="/admin", status_code=status.HTTP_302_FOUND)
        response.set_cookie(key="session_token", value=session_token, httponly=True)
        return response
    else:
        return RedirectResponse(url="/login?error=1", status_code=status.HTTP_302_FOUND)

@app.get("/logout")
async def logout(request: Request):
    session_token = request.cookies.get("session_token")
    if session_token and session_token in SESSIONS:
        del SESSIONS[session_token]
    
    response = RedirectResponse(url="/", status_code=status.HTTP_302_FOUND)
    response.delete_cookie(key="session_token")
    return response

@app.get("/admin")
async def admin_page(request: Request, user: str = Depends(require_admin)):
    return FileResponse("ui/admin.html")

@app.get("/api/admin/status")
async def admin_status(request: Request, user: str = Depends(require_admin)):
    return {
        "user": user,
        "sessions_count": len(SESSIONS),
        "backend_url": AZURE_BACKEND_URL
    }

@app.get("/api/admin/sessions")
async def admin_sessions(request: Request, user: str = Depends(require_admin)):
    """Get information about active sessions"""
    sessions_info = []
    for token, data in SESSIONS.items():
        sessions_info.append({
            "token_prefix": token[:8] + "...",
            "username": data["username"],
            "created": "Active"  # In a real implementation, you'd store timestamps
        })
    return {"sessions": sessions_info, "count": len(sessions_info)}

@app.delete("/api/admin/sessions")
async def clear_sessions(request: Request, user: str = Depends(require_admin)):
    """Clear all sessions except current user's session"""
    current_token = request.cookies.get("session_token")
    if current_token and current_token in SESSIONS:
        # Keep only current user's session
        current_session = SESSIONS[current_token]
        SESSIONS.clear()
        SESSIONS[current_token] = current_session
        return {"message": "All other sessions cleared", "remaining_sessions": 1}
    else:
        SESSIONS.clear()
        return {"message": "All sessions cleared", "remaining_sessions": 0}

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