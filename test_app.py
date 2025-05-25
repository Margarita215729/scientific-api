#!/usr/bin/env python3
"""
Test script for Scientific API application.
Tests all major components before deployment.
"""

import sys
import traceback
import asyncio
from typing import Dict, Any

def test_imports():
    """Test all critical imports."""
    print("🔍 Testing imports...")
    
    try:
        # Core dependencies
        import fastapi
        import uvicorn
        import pandas as pd
        import numpy as np
        import requests
        print("✅ Core dependencies imported successfully")
        
        # Optional dependencies
        try:
            import aiohttp
            print("✅ aiohttp available")
        except ImportError:
            print("⚠️  aiohttp not available (optional for production)")
        
        try:
            import httpx
            print("✅ httpx available")
        except ImportError:
            print("⚠️  httpx not available (optional for production)")
        
        # Astronomical libraries
        import astropy
        from astropy.io import fits
        from astropy.cosmology import Planck15
        print("✅ Astropy libraries imported successfully")
        
        # API modules
        from api.astro_catalog_api import router as astro_router
        from api.ads_api import router as ads_router
        from api.heavy_api import router as heavy_router
        print("✅ API modules imported successfully")
        
        # Utils modules
        from utils.astronomy_catalogs_real import AstronomicalDataProcessor
        from utils.data_preprocessor import AstronomicalDataPreprocessor
        print("✅ Utils modules imported successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        traceback.print_exc()
        return False

def test_main_app():
    """Test main FastAPI application."""
    print("\n🔍 Testing main application...")
    
    try:
        from main import app
        print("✅ Main app imported successfully")
        
        # Check if routes are registered
        routes = [route.path for route in app.routes]
        expected_routes = ["/api/astro", "/api/ads", "/api/health"]
        
        for route in expected_routes:
            if any(r.startswith(route) for r in routes):
                print(f"✅ Route {route} registered")
            else:
                print(f"⚠️  Route {route} not found")
        
        return True
        
    except Exception as e:
        print(f"❌ Main app error: {e}")
        traceback.print_exc()
        return False

async def test_data_processor():
    """Test astronomical data processor."""
    print("\n🔍 Testing data processor...")
    
    try:
        from utils.astronomy_catalogs_real import AstronomicalDataProcessor
        
        processor = AstronomicalDataProcessor()
        print("✅ Data processor created successfully")
        
        # Test catalog info
        from utils.astronomy_catalogs_real import get_catalog_info
        catalog_info = await get_catalog_info()
        print(f"✅ Catalog info retrieved: {len(catalog_info)} catalogs")
        
        return True
        
    except Exception as e:
        print(f"❌ Data processor error: {e}")
        traceback.print_exc()
        return False

def test_file_structure():
    """Test required file structure."""
    print("\n🔍 Testing file structure...")
    
    import os
    
    required_files = [
        "main.py",
        "vercel.json",
        "requirements.txt",
        "api/astro_catalog_api.py",
        "api/ads_api.py",
        "api/heavy_api.py",
        "utils/astronomy_catalogs_real.py",
        "utils/data_preprocessor.py",
        "ui/index.html"
    ]
    
    all_exist = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} missing")
            all_exist = False
    
    return all_exist

def test_vercel_config():
    """Test Vercel configuration."""
    print("\n🔍 Testing Vercel configuration...")
    
    try:
        import json
        
        with open("vercel.json", "r") as f:
            config = json.load(f)
        
        # Check required fields
        required_fields = ["version", "builds", "routes"]
        for field in required_fields:
            if field in config:
                print(f"✅ {field} configured")
            else:
                print(f"❌ {field} missing")
                return False
        
        # Check Python build
        python_build = any(
            build.get("use") == "@vercel/python" 
            for build in config.get("builds", [])
        )
        
        if python_build:
            print("✅ Python build configured")
        else:
            print("❌ Python build not configured")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Vercel config error: {e}")
        return False

async def run_all_tests():
    """Run all tests."""
    print("🚀 Starting Scientific API tests...\n")
    
    tests = [
        ("Imports", test_imports),
        ("Main App", test_main_app),
        ("File Structure", test_file_structure),
        ("Vercel Config", test_vercel_config),
        ("Data Processor", test_data_processor)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "="*50)
    print("📊 TEST SUMMARY")
    print("="*50)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20} {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Ready for deployment.")
        return True
    else:
        print(f"\n⚠️  {total - passed} tests failed. Please fix before deployment.")
        return False

if __name__ == "__main__":
    try:
        success = asyncio.run(run_all_tests())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⏹️  Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n💥 Unexpected error: {e}")
        traceback.print_exc()
        sys.exit(1) 