"""
Security module for the Scientific API.
Provides authentication, rate limiting, input validation, and security headers.
"""

from fastapi import HTTPException, Request, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
import os
import time
import hashlib
import logging
from typing import Dict, Optional, Any
from datetime import datetime, timedelta
import asyncio
from collections import defaultdict
import re

logger = logging.getLogger(__name__)

# Security configuration
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))  # requests per window
RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "3600"))    # window in seconds (1 hour)
API_KEY_HEADER = "X-API-Key"
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "")
USER_API_KEYS = os.getenv("USER_API_KEYS", "").split(",") if os.getenv("USER_API_KEYS") else []

# Rate limiting storage (in production, use Redis)
rate_limit_storage: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"requests": 0, "window_start": time.time()})

# Security bearer scheme
security = HTTPBearer(auto_error=False)

class SecurityHeaders:
    """Add security headers to responses."""
    
    @staticmethod
    def add_security_headers(response: JSONResponse) -> JSONResponse:
        """Add security headers to response."""
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        return response

class RateLimiter:
    """Rate limiting implementation."""
    
    @staticmethod
    def get_client_id(request: Request) -> str:
        """Get unique client identifier."""
        # Try to get API key first
        api_key = request.headers.get(API_KEY_HEADER)
        if api_key:
            return f"api_key_{hashlib.sha256(api_key.encode()).hexdigest()[:16]}"
        
        # Fallback to IP address
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            client_ip = forwarded_for.split(",")[0].strip()
        else:
            client_ip = request.client.host if request.client else "unknown"
        
        return f"ip_{client_ip}"
    
    @staticmethod
    def is_rate_limited(client_id: str) -> bool:
        """Check if client is rate limited."""
        current_time = time.time()
        client_data = rate_limit_storage[client_id]
        
        # Reset window if expired
        if current_time - client_data["window_start"] > RATE_LIMIT_WINDOW:
            client_data["requests"] = 0
            client_data["window_start"] = current_time
        
        # Check if limit exceeded
        if client_data["requests"] >= RATE_LIMIT_REQUESTS:
            return True
        
        # Increment counter
        client_data["requests"] += 1
        return False
    
    @staticmethod
    def get_rate_limit_info(client_id: str) -> Dict[str, Any]:
        """Get rate limit information for client."""
        client_data = rate_limit_storage[client_id]
        current_time = time.time()
        
        # Calculate remaining requests and reset time
        window_remaining = RATE_LIMIT_WINDOW - (current_time - client_data["window_start"])
        requests_remaining = max(0, RATE_LIMIT_REQUESTS - client_data["requests"])
        
        return {
            "requests_remaining": requests_remaining,
            "requests_limit": RATE_LIMIT_REQUESTS,
            "window_remaining_seconds": int(window_remaining),
            "window_size_seconds": RATE_LIMIT_WINDOW
        }

class InputValidator:
    """Input validation and sanitization."""
    
    # Regex patterns for validation
    PATTERNS = {
        "arxiv_id": re.compile(r"^\d{4}\.\d{4,5}(v\d+)?$"),
        "doi": re.compile(r"^10\.\d{4,}/[-._;()/:\w\[\]]+$"),
        "email": re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"),
        "bibcode": re.compile(r"^\d{4}[A-Za-z&.]{5}[A-Za-z0-9.]{13}[A-Z]?$"),
        "coordinates": re.compile(r"^-?\d+\.?\d*$"),
        "safe_string": re.compile(r"^[a-zA-Z0-9\s\-_.,:;()]+$")
    }
    
    @staticmethod
    def validate_query_string(query: str, max_length: int = 500) -> str:
        """Validate and sanitize search query."""
        if not query or not isinstance(query, str):
            raise HTTPException(status_code=400, detail="Query must be a non-empty string")
        
        query = query.strip()
        if len(query) > max_length:
            raise HTTPException(status_code=400, detail=f"Query too long (max {max_length} characters)")
        
        # Remove potentially dangerous characters
        dangerous_chars = ["<", ">", "&", "\"", "'", ";", "|", "&", "$"]
        for char in dangerous_chars:
            query = query.replace(char, "")
        
        return query
    
    @staticmethod
    def validate_arxiv_id(arxiv_id: str) -> str:
        """Validate ArXiv ID format."""
        if not InputValidator.PATTERNS["arxiv_id"].match(arxiv_id):
            raise HTTPException(status_code=400, detail="Invalid ArXiv ID format")
        return arxiv_id
    
    @staticmethod
    def validate_coordinates(ra: float, dec: float) -> tuple:
        """Validate astronomical coordinates."""
        if not (-360 <= ra <= 360):
            raise HTTPException(status_code=400, detail="RA must be between -360 and 360 degrees")
        if not (-90 <= dec <= 90):
            raise HTTPException(status_code=400, detail="DEC must be between -90 and 90 degrees")
        return ra, dec
    
    @staticmethod
    def validate_limit(limit: int, max_limit: int = 1000) -> int:
        """Validate result limit parameter."""
        if limit < 1:
            raise HTTPException(status_code=400, detail="Limit must be at least 1")
        if limit > max_limit:
            raise HTTPException(status_code=400, detail=f"Limit cannot exceed {max_limit}")
        return limit
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename for safe file operations."""
        # Remove path traversal attempts
        filename = os.path.basename(filename)
        
        # Remove dangerous characters
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        
        # Limit length
        if len(filename) > 255:
            name, ext = os.path.splitext(filename)
            filename = name[:250] + ext
        
        return filename

class APIKeyAuth:
    """API Key authentication."""
    
    @staticmethod
    def verify_api_key(api_key: str) -> Dict[str, Any]:
        """Verify API key and return user info."""
        if not api_key:
            return {"valid": False, "user_type": "anonymous", "permissions": ["read"]}
        
        # Check admin key
        if ADMIN_API_KEY and api_key == ADMIN_API_KEY:
            return {
                "valid": True,
                "user_type": "admin",
                "permissions": ["read", "write", "admin", "ml_training", "data_management"]
            }
        
        # Check user keys
        if api_key in USER_API_KEYS:
            return {
                "valid": True,
                "user_type": "user",
                "permissions": ["read", "write", "ml_training"]
            }
        
        # Invalid key
        return {"valid": False, "user_type": "invalid", "permissions": []}

# Dependency functions for FastAPI

async def rate_limit_check(request: Request):
    """Rate limiting dependency."""
    client_id = RateLimiter.get_client_id(request)
    
    if RateLimiter.is_rate_limited(client_id):
        rate_info = RateLimiter.get_rate_limit_info(client_id)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded",
            headers={
                "X-RateLimit-Limit": str(RATE_LIMIT_REQUESTS),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(int(time.time() + rate_info["window_remaining_seconds"])),
                "Retry-After": str(rate_info["window_remaining_seconds"])
            }
        )
    
    return client_id

async def get_current_user(request: Request, credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current user from API key."""
    api_key = None
    
    # Try Authorization header first
    if credentials:
        api_key = credentials.credentials
    
    # Try X-API-Key header
    if not api_key:
        api_key = request.headers.get(API_KEY_HEADER)
    
    user_info = APIKeyAuth.verify_api_key(api_key)
    
    return {
        "api_key": api_key,
        "user_type": user_info["user_type"],
        "permissions": user_info["permissions"],
        "is_authenticated": user_info["valid"]
    }

async def require_authentication(current_user: dict = Depends(get_current_user)):
    """Require valid authentication."""
    if not current_user["is_authenticated"]:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Valid API key required",
            headers={"WWW-Authenticate": "Bearer"}
        )
    return current_user

async def require_admin(current_user: dict = Depends(require_authentication)):
    """Require admin privileges."""
    if "admin" not in current_user["permissions"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required"
        )
    return current_user

async def require_ml_access(current_user: dict = Depends(require_authentication)):
    """Require ML training access."""
    if "ml_training" not in current_user["permissions"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="ML training access required"
        )
    return current_user

# Utility functions

def log_security_event(event_type: str, client_id: str, details: Dict[str, Any] = None):
    """Log security-related events."""
    log_data = {
        "event_type": event_type,
        "client_id": client_id,
        "timestamp": datetime.utcnow().isoformat(),
        "details": details or {}
    }
    
    logger.warning(f"SECURITY_EVENT: {log_data}")

def create_secure_response(data: Any, status_code: int = 200) -> JSONResponse:
    """Create response with security headers."""
    response = JSONResponse(content=data, status_code=status_code)
    return SecurityHeaders.add_security_headers(response)

# Middleware for automatic security headers
class SecurityMiddleware:
    """Middleware to add security headers to all responses."""
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            async def send_wrapper(message):
                if message["type"] == "http.response.start":
                    headers = dict(message.get("headers", []))
                    
                    # Add security headers
                    security_headers = {
                        b"x-content-type-options": b"nosniff",
                        b"x-frame-options": b"DENY",
                        b"x-xss-protection": b"1; mode=block",
                        b"strict-transport-security": b"max-age=31536000; includeSubDomains",
                        b"referrer-policy": b"strict-origin-when-cross-origin"
                    }
                    
                    for key, value in security_headers.items():
                        if key not in headers:
                            headers[key] = value
                    
                    message["headers"] = list(headers.items())
                
                await send(message)
            
            await self.app(scope, receive, send_wrapper)
        else:
            await self.app(scope, receive, send)

# Health check for security components
def get_security_status() -> Dict[str, Any]:
    """Get security component status."""
    return {
        "rate_limiting": {
            "enabled": True,
            "requests_per_window": RATE_LIMIT_REQUESTS,
            "window_seconds": RATE_LIMIT_WINDOW,
            "active_clients": len(rate_limit_storage)
        },
        "authentication": {
            "admin_key_configured": bool(ADMIN_API_KEY),
            "user_keys_count": len(USER_API_KEYS),
            "api_key_header": API_KEY_HEADER
        },
        "input_validation": {
            "enabled": True,
            "patterns_count": len(InputValidator.PATTERNS)
        },
        "security_headers": {
            "enabled": True
        }
    }
