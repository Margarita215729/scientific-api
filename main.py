from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import os

app = FastAPI(title="Irina Vinokur - Artist Portfolio")

# Mount static files from ui directory
app.mount("/static", StaticFiles(directory="ui"), name="static")

# Health check endpoint
@app.get("/ping")
async def ping():
    return {"status": "ok", "service": "artist-portfolio"}

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "service": "artist-portfolio"}

# Static file serving
@app.get("/")
async def serve_index():
    return FileResponse("ui/index.html")

@app.get("/portfolio")
async def serve_portfolio():
    return FileResponse("ui/portfolio.html")

@app.get("/about")
async def serve_about():
    return FileResponse("ui/about.html")

@app.get("/contact")
async def serve_contact():
    return FileResponse("ui/contact.html")

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