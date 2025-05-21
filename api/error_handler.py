"""
Error handling module for the FastAPI application.
"""

import logging
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
import sys
import traceback

# Configure logger
logger = logging.getLogger(__name__)

def setup_error_handlers(app: FastAPI):
    """
    Set up global error handlers for the FastAPI application.
    This improves error reporting and ensures we don't expose
    sensitive information in error responses.
    """
    
    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        logger.error(f"HTTP exception: {exc.detail} (status_code={exc.status_code})")
        return JSONResponse(
            status_code=exc.status_code,
            content={"message": exc.detail}
        )
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        logger.error(f"Validation error: {exc}")
        return JSONResponse(
            status_code=422,
            content={"message": "Invalid request parameters", "details": exc.errors()}
        )
        
    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        """Handle any unhandled exceptions"""
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        
        # Get exception traceback
        trace = traceback.format_exception(type(exc), exc, exc.__traceback__)
        trace_str = "".join(trace)
        
        # Log the full traceback
        logger.error(f"Exception traceback:\n{trace_str}")
        
        # In production, don't expose the full traceback to the client
        if "VERCEL" in sys.modules or "AWS_LAMBDA_FUNCTION_NAME" in sys.modules:
            # We're in a production environment
            return JSONResponse(
                status_code=500,
                content={"message": "Internal server error", "error_type": str(type(exc).__name__)}
            )
        else:
            # We're in a development environment, so return more details
            return JSONResponse(
                status_code=500,
                content={
                    "message": "Internal server error",
                    "error": str(exc),
                    "error_type": str(type(exc).__name__),
                    "traceback": trace_str
                }
            )
    
    logger.info("Error handlers have been set up")
    return app 