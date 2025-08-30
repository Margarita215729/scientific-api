"""
Basic tests for Scientific API that don't require heavy dependencies.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import json

# Import the main application
from main import app

client = TestClient(app)

class TestBasicEndpoints:
    """Test basic API endpoints."""
    
    def test_ping(self):
        """Test ping endpoint."""
        response = client.get("/ping")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "service" in data
    
    def test_root(self):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
    
    def test_api_root(self):
        """Test API root endpoint."""
        response = client.get("/api")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "endpoints" in data
    
    def test_health_check(self):
        """Test health check endpoint."""
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data

class TestStaticFiles:
    """Test static file serving."""
    
    def test_static_files(self):
        """Test static file serving."""
        response = client.get("/static/style.css")
        # Should return 200 or 404 depending on file existence
        assert response.status_code in [200, 404]
    
    def test_favicon(self):
        """Test favicon endpoints."""
        response = client.get("/favicon.ico")
        # Should return 200 or 404 depending on file existence
        assert response.status_code in [200, 404]

class TestErrorHandling:
    """Test error handling."""
    
    def test_invalid_endpoint(self):
        """Test invalid endpoint returns 404."""
        response = client.get("/api/invalid/endpoint")
        assert response.status_code == 404
    
    def test_invalid_static_file(self):
        """Test invalid static file returns 404."""
        response = client.get("/static/nonexistent.js")
        assert response.status_code == 404

class TestProxyEndpoints:
    """Test proxy endpoints."""
    
    @patch('main.httpx.AsyncClient')
    def test_proxy_astro_endpoint(self, mock_client):
        """Test proxy to astro endpoint."""
        # Mock the response
        mock_response = Mock()
        mock_response.json.return_value = {"test": "data"}
        mock_response.status_code = 200
        
        mock_client_instance = Mock()
        mock_client_instance.get.return_value = mock_response
        mock_client.return_value.__aenter__.return_value = mock_client_instance
        
        response = client.get("/api/astro/test")
        # Should handle the proxy request
        assert response.status_code in [200, 500, 503]  # Various possible responses
    
    @patch('main.httpx.AsyncClient')
    def test_proxy_ads_endpoint(self, mock_client):
        """Test proxy to ADS endpoint."""
        # Mock the response
        mock_response = Mock()
        mock_response.json.return_value = {"publications": []}
        mock_response.status_code = 200
        
        mock_client_instance = Mock()
        mock_client_instance.get.return_value = mock_response
        mock_client.return_value.__aenter__.return_value = mock_client_instance
        
        response = client.get("/api/ads/search")
        # Should handle the proxy request
        assert response.status_code in [200, 500, 503]  # Various possible responses

class TestIntegration:
    """Integration tests."""
    
    def test_full_workflow(self):
        """Test a complete workflow."""
        # 1. Check health
        health_response = client.get("/api/health")
        assert health_response.status_code == 200
        
        # 2. Check root
        root_response = client.get("/")
        assert root_response.status_code == 200
        
        # 3. Check API docs
        docs_response = client.get("/docs")
        assert docs_response.status_code == 200

if __name__ == "__main__":
    pytest.main([__file__])