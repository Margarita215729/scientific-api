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

    # Check dependencies
    try:
        from api.dependencies import print_dependency_status
        print_dependency_status()
    except Exception as e:
        logger.error(f"Error checking dependencies: {e}", exc_info=True)

    # Import simple router that doesn't depend on pandas
    try:
        from api.simple_router import router as simple_router
        app.include_router(simple_router, prefix="/astro", tags=["Simple Astronomical Data"])
        logger.info("Successfully loaded simple_router")
    except Exception as e:
        logger.error(f"Error loading simple_router: {e}", exc_info=True)
        # Create a basic router to ensure minimal functionality
        from fastapi import APIRouter
        simple_router = APIRouter()
        
        @simple_router.get("/status")
        async def fallback_status():
            return {
                "status": "limited",
                "message": "Running in fallback mode due to initialization errors",
                "error": str(e)
            }
        
        app.include_router(simple_router, prefix="/astro", tags=["Simple Astronomical Data (Fallback)"])

    # Try to import routers, but don't fail if they're not available
    try:
        from api.astro_catalog_api import router as astro_router
        app.include_router(astro_router, prefix="/astro/full", tags=["Astronomical Catalogs"])
        logger.info("Successfully loaded full astro_router")
    except ImportError as e:
        logger.warning(f"Could not import astro_router: {e}")
    except Exception as e:
        logger.error(f"Unexpected error loading astro_router: {e}", exc_info=True)

    try:
        from api.ads_api import router as ads_router
        app.include_router(ads_router, prefix="/ads", tags=["ADS Literature"])
        logger.info("Successfully loaded ads_router")
    except ImportError as e:
        logger.warning(f"Could not import ads_router: {e}")
    except Exception as e:
        logger.error(f"Unexpected error loading ads_router: {e}", exc_info=True)

    try:
        from api.dataset_api import router as dataset_router
        app.include_router(dataset_router, prefix="/datasets", tags=["Datasets"])
        logger.info("Successfully loaded dataset_router")
    except ImportError as e:
        logger.warning(f"Could not import dataset_router: {e}")
    except Exception as e:
        logger.error(f"Unexpected error loading dataset_router: {e}", exc_info=True)

    try:
        from api.file_manager_api import router as file_router
        app.include_router(file_router, prefix="/files", tags=["File Management"])
        logger.info("Successfully loaded file_router")
    except ImportError as e:
        logger.warning(f"Could not import file_router: {e}")
    except Exception as e:
        logger.error(f"Unexpected error loading file_router: {e}", exc_info=True)

    try:
        from api.ml_models import router as ml_router
        app.include_router(ml_router, prefix="/ml", tags=["Machine Learning Models"])
        logger.info("Successfully loaded ml_router")
    except ImportError as e:
        logger.warning(f"Could not import ml_router: {e}")
    except Exception as e:
        logger.error(f"Unexpected error loading ml_router: {e}", exc_info=True)

    @app.get("/", response_class=HTMLResponse)
    async def root(request: Request):
        try:
            return templates.TemplateResponse("index.html", {"request": request})
        except Exception as e:
            logger.error(f"Error rendering index.html: {e}", exc_info=True)
            return JSONResponse(
                status_code=500,
                content={"message": "Error rendering page", "error": str(e)}
            )

    @app.get("/ads")
    async def ads_page(request: Request):
        try:
            return templates.TemplateResponse("ads.html", {"request": request})
        except Exception as e:
            logger.error(f"Error rendering ads.html: {e}", exc_info=True)
            return JSONResponse(
                status_code=500,
                content={"message": "Error rendering page", "error": str(e)}
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
                "astro": "/astro/...",
                "ads": "/ads/...",
                "datasets": "/datasets/...",
                "files": "/files/...",
                "ml": "/ml/..."
            }
        }

    @app.get("/ping")
    async def ping():
        import datetime
        return {
            "status": "ok", 
            "message": "API is up and running",
            "python_version": PYTHON_VERSION,
            "timestamp": datetime.datetime.now().isoformat()
        }

    @app.get("/health")
    async def health():
        """
        Health check endpoint for monitoring systems.
        """
        import datetime
        from api.dependencies import get_dependency_status
        
        try:
            # Check if critical dependencies are available
            deps = get_dependency_status()
            critical_deps = ["pandas", "numpy", "requests"]
            missing_critical = [dep for dep in critical_deps if deps.get(dep, {}).get("available") is False]
            
            status = "ok" 
            if missing_critical:
                status = "degraded"
                message = f"Missing critical dependencies: {', '.join(missing_critical)}"
            else:
                message = "All systems operational"
                
            return {
                "status": status,
                "message": message,
                "python_version": PYTHON_VERSION,
                "timestamp": datetime.datetime.now().isoformat(),
                "dependencies": {name: info["available"] for name, info in deps.items()}
            }
        except Exception as e:
            logger.error(f"Health check failed: {e}", exc_info=True)
            return {
                "status": "error",
                "message": f"Health check failed: {str(e)}",
                "timestamp": datetime.datetime.now().isoformat()
            }

except Exception as e:
    # Get traceback
    tb = traceback.format_exception(type(e), e, e.__traceback__)
    tb_str = "".join(tb)
    
    logger.critical(f"Error initializing application: {e}", exc_info=True)
    logger.critical(f"Traceback: {tb_str}")
    
    # Get Python version (from env_info if available, otherwise calculate it)
    python_version = env_info.get("python_version") if "env_info" in locals() else f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    
    # Create minimal app to show error
    app = FastAPI(title="Scientific API (Error)")
    
    @app.get("/")
    async def error_root():
        return JSONResponse(
            status_code=500,
            content={
                "message": "Application failed to initialize", 
                "error": str(e),
                "python_version": python_version
            }
        )
    
    @app.get("/ping")
    async def error_ping():
        import datetime
        return {
            "status": "error", 
            "message": "Application failed to initialize", 
            "error": str(e),
            "python_version": python_version,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
    @app.get("/debug")
    async def error_debug():
        import os
        import datetime
        if os.environ.get("VERCEL") != "1":  # Only show in non-production
            return JSONResponse(
                status_code=500,
                content={
                    "message": "Application failed to initialize", 
                    "error": str(e),
                    "traceback": tb_str,
                    "python_version": python_version,
                    "timestamp": datetime.datetime.now().isoformat()
                }
            )
        else:
            return JSONResponse(
                status_code=403,
                content={"message": "Debug endpoint not available in production"}
            ) 