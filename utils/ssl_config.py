"""
SSL configuration module for fixing certificate issues.
"""

import os
import ssl
import requests
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def configure_ssl_for_requests() -> requests.Session:
    """Create a properly configured requests session with SSL."""
    session = requests.Session()
    
    try:
        # Try to use system certificates first
        import certifi
        cert_path = certifi.where()
        
        # Check if the certificate file actually exists
        if os.path.exists(cert_path):
            session.verify = cert_path
            logger.info(f"Using SSL certificates from: {cert_path}")
        else:
            logger.warning(f"Certificate file not found at {cert_path}, using system default")
            session.verify = True
            
    except ImportError:
        logger.warning("certifi not available, using system SSL")
        session.verify = True
    except Exception as e:
        logger.error(f"SSL configuration error: {e}")
        # Fallback to no verification for development
        session.verify = False
        try:
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        except ImportError:
            pass
    
    return session

def create_ssl_context() -> ssl.SSLContext:
    """Create a properly configured SSL context."""
    try:
        context = ssl.create_default_context()
        
        # Try to load system certificates
        try:
            import certifi
            cert_file = certifi.where()
            if os.path.exists(cert_file):
                context.load_verify_locations(cert_file)
                logger.info("SSL context configured with certifi certificates")
            else:
                logger.warning("certifi certificate file not found, using system default")
        except ImportError:
            logger.info("Using system default SSL context")
        
        return context
        
    except Exception as e:
        logger.error(f"Error creating SSL context: {e}")
        # Return unverified context as fallback
        context = ssl.create_default_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        return context

def test_ssl_connectivity(url: str) -> bool:
    """Test SSL connectivity to a URL."""
    try:
        session = configure_ssl_for_requests()
        response = session.get(url, timeout=10)
        return response.status_code == 200
    except Exception as e:
        logger.error(f"SSL connectivity test failed for {url}: {e}")
        return False

# Environment variable to disable SSL verification in development
if os.getenv("DISABLE_SSL_VERIFICATION", "false").lower() == "true":
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    # Monkey patch requests to disable SSL verification
    import requests.adapters
    original_send = requests.adapters.HTTPAdapter.send
    
    def send_no_verify(self, request, *args, **kwargs):
        kwargs['verify'] = False
        return original_send(self, request, *args, **kwargs)
    
    requests.adapters.HTTPAdapter.send = send_no_verify
    logger.warning("SSL verification disabled via environment variable")

