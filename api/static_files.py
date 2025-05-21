from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import pathlib
import os

def setup_static_files(app: FastAPI):
    """
    Set up static files based on the environment.
    For Vercel, we use a special configuration to serve UI files.
    """
    # Get the directory of the current file
    current_dir = pathlib.Path(__file__).parent.parent
    ui_dir = current_dir / "ui"
    
    # Check if we're running on Vercel
    if os.environ.get("VERCEL") == "1":
        # On Vercel, static files are served via routes configuration
        # But we still need to mount them for FastAPI's reference
        print("Running on Vercel - static files will be served via Vercel's routing")
    else:
        # Local development - mount static files directly
        print(f"Setting up static files from {ui_dir}")
        app.mount("/static", StaticFiles(directory=str(ui_dir)), name="static")
        
    return ui_dir 