"""
Backup/fallback app for Vercel deployment.
This is a minimal version of the app that doesn't depend on pandas or other heavy libraries.
"""

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pathlib

# Initialize FastAPI app
app = FastAPI(title="Scientific API (Minimal)")

# Get the directory of the current file
current_dir = pathlib.Path(__file__).parent.parent
ui_dir = current_dir / "ui"

# Mount static files
app.mount("/static", StaticFiles(directory=str(ui_dir)), name="static")

# Set up Jinja2 templates
templates = Jinja2Templates(directory=str(ui_dir))

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/ads")
async def ads_page(request: Request):
    return templates.TemplateResponse("ads.html", {"request": request})

@app.get("/ping")
async def ping():
    return {"status": "ok", "message": "Minimal API is up and running"}

@app.get("/astro/status")
async def get_status():
    """
    Return simple status information for the API.
    """
    return {
        "status": "ok",
        "catalogs": [
            {"name": "SDSS DR17", "description": "Spectroscopic catalog", "available": False},
            {"name": "Euclid Q1", "description": "MER Final catalog", "available": False},
            {"name": "DESI DR1", "description": "ELG clustering catalog", "available": False},
            {"name": "DES Y6", "description": "Gold catalog", "available": False}
        ],
        "message": "This is a minimal version of the API. For full functionality, ensure all dependencies are installed."
    }

@app.get("/astro/galaxies")
async def get_galaxies():
    """
    Return sample galaxy data.
    """
    # Sample data that doesn't require pandas
    sample_galaxies = [
        {"RA": 150.1, "DEC": 2.2, "redshift": 0.5, "source": "SDSS", "X": 100.0, "Y": 200.0, "Z": 300.0},
        {"RA": 151.2, "DEC": 2.3, "redshift": 0.6, "source": "SDSS", "X": 110.0, "Y": 210.0, "Z": 310.0},
        {"RA": 149.3, "DEC": 2.1, "redshift": 0.4, "source": "Euclid", "X": 90.0, "Y": 190.0, "Z": 290.0},
        {"RA": 152.4, "DEC": 2.4, "redshift": 0.7, "source": "DESI", "X": 120.0, "Y": 220.0, "Z": 320.0},
        {"RA": 153.5, "DEC": 2.5, "redshift": 0.8, "source": "DESI", "X": 130.0, "Y": 230.0, "Z": 330.0},
    ]
    
    return {
        "count": len(sample_galaxies),
        "galaxies": sample_galaxies
    }

@app.get("/astro/statistics")
async def get_statistics():
    """
    Return sample statistics for the API.
    """
    return {
        "total_galaxies": 10000,
        "redshift": {
            "min": 0.1,
            "max": 1.0,
            "mean": 0.55
        },
        "sources": {
            "SDSS": 2500,
            "Euclid": 3000,
            "DESI": 2000,
            "DES": 2500
        }
    } 