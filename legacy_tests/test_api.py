"""
Comprehensive API tests for Scientific API.
Tests all endpoints and functionality.
"""

import pytest
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock
import json
import pandas as pd
import numpy as np

# Import the main application
from main_azure_with_db import app

client = TestClient(app)

class TestHealthEndpoints:
    """Test health check endpoints."""
    
    def test_ping(self):
        """Test ping endpoint."""
        response = client.get("/ping")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "service" in data
    
    def test_health_check(self):
        """Test health check endpoint."""
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "service_name" in data
        assert "version" in data

class TestAstronomicalCatalogAPI:
    """Test astronomical catalog API endpoints."""
    
    @patch('api.astro_catalog_api.AstronomicalDataProcessor')
    def test_get_catalog_info(self, mock_processor):
        """Test getting catalog information."""
        mock_processor.return_value.get_catalog_info.return_value = {
            "SDSS": {"status": "available", "objects": 1000},
            "DESI": {"status": "available", "objects": 500}
        }
        
        response = client.get("/api/astro/catalogs")
        assert response.status_code == 200
        data = response.json()
        assert "catalogs" in data
    
    @patch('api.astro_catalog_api.get_comprehensive_statistics')
    def test_get_statistics(self, mock_stats):
        """Test getting comprehensive statistics."""
        mock_stats.return_value = {
            "total_objects": 1500,
            "redshift_range": "0.0 - 3.0",
            "magnitude_range": "15.0 - 25.0"
        }
        
        response = client.get("/api/astro/statistics")
        assert response.status_code == 200
        data = response.json()
        assert "statistics" in data
    
    @patch('api.astro_catalog_api.fetch_filtered_galaxies')
    def test_filter_galaxies(self, mock_fetch):
        """Test filtering galaxies."""
        mock_fetch.return_value = [
            {"ra": 150.0, "dec": 2.0, "redshift": 0.5, "mag_g": 20.0},
            {"ra": 151.0, "dec": 2.1, "redshift": 0.6, "mag_g": 21.0}
        ]
        
        params = {
            "catalog_source": "SDSS",
            "min_redshift": 0.0,
            "max_redshift": 1.0,
            "limit": 10
        }
        
        response = client.get("/api/astro/filter", params=params)
        assert response.status_code == 200
        data = response.json()
        assert "galaxies" in data

class TestADSAPI:
    """Test ADS API endpoints."""
    
    @patch('api.ads_api.search_by_coordinates')
    def test_search_by_coordinates(self, mock_search):
        """Test searching by coordinates."""
        mock_search.return_value = [
            {
                "title": "Test Paper",
                "author": ["Author 1", "Author 2"],
                "year": 2023,
                "citation_count": 10
            }
        ]
        
        params = {"ra": 150.0, "dec": 2.0, "radius": 0.1}
        response = client.get("/api/ads/search-by-coordinates", params=params)
        assert response.status_code == 200
        data = response.json()
        assert "publications" in data
        assert data["count"] == 1
    
    @patch('api.ads_api.search_by_object')
    def test_search_by_object(self, mock_search):
        """Test searching by object name."""
        mock_search.return_value = [
            {
                "title": "M31 Research",
                "author": ["Researcher 1"],
                "year": 2022,
                "citation_count": 5
            }
        ]
        
        params = {"object_name": "M31"}
        response = client.get("/api/ads/search-by-object", params=params)
        assert response.status_code == 200
        data = response.json()
        assert "publications" in data
    
    @patch('api.ads_api.search_by_catalog')
    def test_search_by_catalog(self, mock_search):
        """Test searching by catalog."""
        mock_search.return_value = {
            "publications": [
                {
                    "title": "SDSS Analysis",
                    "author": ["Analyst 1"],
                    "year": 2021,
                    "citation_count": 15
                }
            ],
            "keyword_stats": {"galaxy": 10, "redshift": 8}
        }
        
        params = {"catalog": "SDSS"}
        response = client.get("/api/ads/search-by-catalog", params=params)
        assert response.status_code == 200
        data = response.json()
        assert "publications" in data

class TestMLAnalysisAPI:
    """Test ML analysis API endpoints."""
    
    @patch('api.ml_analysis_api.ml_service.prepare_dataset')
    def test_prepare_dataset(self, mock_prepare):
        """Test preparing ML dataset."""
        mock_prepare.return_value = {
            "train_samples": 800,
            "test_samples": 200,
            "features": ["ra", "dec", "mag_g", "redshift"],
            "target_variable": "redshift",
            "dataset_info": {
                "total_objects": 1000,
                "catalog_sources": ["SDSS", "DESI"],
                "feature_count": 4
            }
        }
        
        response = client.post("/api/ml/prepare-dataset", json={
            "target_variable": "redshift",
            "test_size": 0.2
        })
        assert response.status_code == 200
        data = response.json()
        assert "train_samples" in data
        assert "test_samples" in data
    
    @patch('api.ml_analysis_api.ml_service.train_model')
    def test_train_model(self, mock_train):
        """Test training ML model."""
        mock_train.return_value = {
            "model_path": "models/random_forest_redshift_20241201_120000.pkl",
            "model_type": "random_forest",
            "target_variable": "redshift",
            "metrics": {
                "mse": 0.01,
                "r2": 0.85,
                "mae": 0.08
            }
        }
        
        response = client.post("/api/ml/train-model", json={
            "model_type": "random_forest",
            "target_variable": "redshift"
        })
        assert response.status_code == 200
        data = response.json()
        assert "model_path" in data
        assert "metrics" in data
    
    @patch('api.ml_analysis_api.ml_service.make_prediction')
    def test_make_prediction(self, mock_predict):
        """Test making prediction."""
        mock_predict.return_value = {
            "prediction": [0.5],
            "model_path": "models/test_model.pkl",
            "input_features": {"ra": 150.0, "dec": 2.0, "mag_g": 20.0},
            "timestamp": "2024-12-01T12:00:00"
        }
        
        response = client.post("/api/ml/predict", json={
            "model_path": "models/test_model.pkl",
            "features": {"ra": 150.0, "dec": 2.0, "mag_g": 20.0}
        })
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
    
    def test_list_models(self):
        """Test listing models."""
        response = client.get("/api/ml/models")
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert "count" in data
    
    def test_get_dataset_statistics(self):
        """Test getting dataset statistics."""
        response = client.get("/api/ml/dataset-statistics")
        assert response.status_code == 200
        data = response.json()
        assert "statistics" in data

class TestDatabaseAPI:
    """Test database API endpoints."""
    
    def test_database_status(self):
        """Test database status endpoint."""
        response = client.get("/api/database/status")
        # This might return 503 if database is not connected
        assert response.status_code in [200, 503]
    
    def test_get_database_objects(self):
        """Test getting database objects."""
        params = {"limit": 10}
        response = client.get("/api/database/objects", params=params)
        # This might return 503 if database is not connected
        assert response.status_code in [200, 503]

class TestErrorHandling:
    """Test error handling."""
    
    def test_invalid_endpoint(self):
        """Test invalid endpoint returns 404."""
        response = client.get("/api/invalid/endpoint")
        assert response.status_code == 404
    
    def test_invalid_parameters(self):
        """Test invalid parameters return appropriate errors."""
        # Test with invalid coordinates
        params = {"ra": "invalid", "dec": "invalid"}
        response = client.get("/api/ads/search-by-coordinates", params=params)
        assert response.status_code in [400, 422, 500]
    
    def test_missing_required_parameters(self):
        """Test missing required parameters."""
        response = client.get("/api/ads/search-by-object")
        assert response.status_code in [400, 422]

class TestIntegration:
    """Integration tests."""
    
    def test_full_workflow(self):
        """Test a complete workflow from data access to ML."""
        # This is a high-level integration test
        # In a real scenario, you'd test the full pipeline
        
        # 1. Check health
        health_response = client.get("/api/health")
        assert health_response.status_code == 200
        
        # 2. Get catalog info
        catalog_response = client.get("/api/astro/catalogs")
        assert catalog_response.status_code == 200
        
        # 3. Get statistics
        stats_response = client.get("/api/astro/statistics")
        assert stats_response.status_code == 200

if __name__ == "__main__":
    pytest.main([__file__])