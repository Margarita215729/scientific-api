"""
Main FastAPI application module.
This module initializes the FastAPI app and sets up all routes and middleware.
"""

import sys
import traceback
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

# Set up logging first
from api.logging_setup import setup_logging, log_environment_info
logger = setup_logging(level="INFO")
env_info = log_environment_info(logger)

try:
    # Import configuration module
    from api.config import (
        IS_PRODUCTION, PYTHON_VERSION, API_TITLE, 
        API_DESCRIPTION, API_VERSION, get_config
    )
    
    # Log startup information
    logger.info(f"Starting Scientific API [Version {API_VERSION}]")
    
    # Initialize FastAPI app
    app = FastAPI(
        title=API_TITLE,
        description=API_DESCRIPTION,
        version=API_VERSION
    )
    
    # Set up error handlers
    try:
        from api.error_handler import setup_error_handlers
        app = setup_error_handlers(app)
    except Exception as e:
        logger.error(f"Failed to set up error handlers: {e}")

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Setup static files and get UI directory
    try:
        from api.static_files import setup_static_files
        ui_dir = setup_static_files(app)
        logger.info(f"Static files setup with UI directory: {ui_dir}")
    except Exception as e:
        logger.error(f"Error setting up static files: {e}")
        # Provide a fallback path for templates
        import pathlib
        ui_dir = pathlib.Path(__file__).parent.parent / "ui"
        logger.info(f"Using fallback UI directory: {ui_dir}")

    # Set up Jinja2 templates
    try:
        templates = Jinja2Templates(directory=str(ui_dir))
        logger.info(f"Jinja2 templates initialized with directory: {ui_dir}")
    except Exception as e:
        logger.error(f"Error setting up Jinja2 templates: {e}")
        raise

    # Import lightweight router
    try:
        from api.heavy_api import router as heavy_router
        app.include_router(heavy_router)
        logger.info("Successfully loaded heavy_router with full functionality")
    except Exception as e:
        logger.error(f"Error loading heavy_router: {e}", exc_info=True)
        # Fallback to light router
        # try:
        #     from api.light_api import router as light_router
        #     app.include_router(light_router)
        #     logger.info("Successfully loaded light_router as fallback")
        # except Exception as e2:
        #     logger.error(f"Error loading light_router fallback: {e2}", exc_info=True)
        #     raise
        raise # Re-raise the exception if heavy_router fails to load

    @app.get("/", response_class=HTMLResponse)
    async def root(request: Request):
        try:
            logger.info(f"Attempting to render index.html from templates directory: {ui_dir}")
            return templates.TemplateResponse("index.html", {"request": request})
        except Exception as e:
            logger.error(f"Error rendering index.html: {e}", exc_info=True)
            return JSONResponse(
                status_code=500,
                content={"message": "Error rendering page", "error": str(e), "ui_dir": str(ui_dir)}
            )

    @app.get("/api")
    async def api_info():
        try:
            from api.dependencies import get_dependency_status
            deps = get_dependency_status()
        except Exception as e:
            logger.error(f"Error getting dependencies status: {e}", exc_info=True)
            deps = {"error": str(e)}
        
        config = get_config()
        return {
            "message": API_TITLE,
            "version": API_VERSION,
            "python_version": PYTHON_VERSION,
            "environment": "Production" if IS_PRODUCTION else "Development",
            "dependencies": deps,
            "endpoints": {
                "basic": {
                    "ping": "/ping",
                    "status": "/status",
                    "docs": "/docs"
                },
                "ads": {
                    "search_by_object": "/ads/search-by-object",
                    "search_by_author": "/ads/search-by-author", 
                    "search_by_title": "/ads/search-by-title",
                    "basic": "/ads/basic"
                },
                "astro": {
                    "status": "/astro/status",
                    "full_catalogs": "/astro/full/*",
                    "galaxies": "/astro/full/galaxies",
                    "stars": "/astro/full/stars",
                    "nebulae": "/astro/full/nebulae"
                },
                "additional": {
                    "datasets": "/datasets/*",
                    "files": "/files/*", 
                    "ml": "/ml/*",
                    "analysis": "/analysis/*"
                }
            }
        }

except Exception as e:
    # Get traceback
    tb = traceback.format_exception(type(e), e, e.__traceback__)
    tb_str = "".join(tb)
    
    logger.critical(f"Error initializing application: {e}", exc_info=True)
    logger.critical(f"Traceback: {tb_str}")
    
    # Create minimal app to show error
    app = FastAPI(title="Scientific API (Error)")
    
    @app.get("/")
    async def error_root():
        return {"error": "Application failed to initialize", "message": str(e)}
    
    @app.get("/ping")
    async def error_ping():
        return {"status": "error", "message": "Application failed to initialize"} 