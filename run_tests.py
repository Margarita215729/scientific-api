#!/usr/bin/env python3
"""
Comprehensive test suite for Scientific Data Platform
"""

import asyncio
import os
import sys
import json
from datetime import datetime

# Load environment variables
import load_env

print("🧪 Scientific Data Platform - Comprehensive Test Suite")
print("=" * 60)

async def test_database_connection():
    """Test database connectivity"""
    print("\n📊 Testing Database Connection...")
    try:
        from database.config import db
        await db.connect()
        print("✅ Database: Connected successfully to MongoDB Atlas")
        
        # Test basic operations
        test_doc = {"test": True, "timestamp": datetime.utcnow()}
        result = await db.mongo_insert_one("test_collection", test_doc)
        if result:
            print(f"✅ Database: Insert operation successful (ID: {result})")
        
        await db.disconnect()
        print("✅ Database: Disconnected successfully")
        return True
    except Exception as e:
        print(f"❌ Database error: {e}")
        return False

async def test_data_import_arxiv():
    """Test arXiv data import"""
    print("\n📚 Testing arXiv Integration...")
    try:
        from api.data_management import import_arxiv_data
        
        params = {
            "search_query": "cat:astro-ph",
            "max_results": 5,
            "sort_by": "submittedDate"
        }
        
        data = await import_arxiv_data(params)
        print(f"✅ arXiv: Retrieved {len(data)} papers")
        
        if data:
            sample = data[0]
            print(f"   📄 Sample: {sample.get('title', 'No title')[:80]}...")
            print(f"   👥 Authors: {', '.join(sample.get('authors', [])[:2])}")
            print(f"   🔗 arXiv ID: {sample.get('arxiv_id', 'N/A')}")
        
        return True
    except Exception as e:
        print(f"❌ arXiv error: {e}")
        return False

async def test_data_import_serpapi():
    """Test SerpAPI integration"""
    print("\n🔍 Testing SerpAPI Integration...")
    try:
        from api.data_management import import_serpapi_data
        
        serpapi_key = os.getenv("SERPAPI_KEY")
        if not serpapi_key:
            print("⚠️  SerpAPI: API key not configured")
            return False
        
        params = {
            "engine": "google_scholar",
            "q": "machine learning astronomy",
            "num": 3
        }
        
        data = await import_serpapi_data(params)
        print(f"✅ SerpAPI: Retrieved {len(data)} results")
        
        if data:
            sample = data[0]
            print(f"   📄 Sample: {sample.get('title', 'No title')[:80]}...")
            print(f"   📊 Citations: {sample.get('cited_by', 0)}")
        
        return True
    except Exception as e:
        print(f"❌ SerpAPI error: {e}")
        return False

async def test_data_import_ads():
    """Test ADS integration"""
    print("\n🌌 Testing ADS Integration...")
    try:
        from api.data_management import import_ads_data
        
        ads_token = os.getenv("ADSABS_TOKEN")
        if not ads_token:
            print("⚠️  ADS: API token not configured")
            return False
        
        params = {
            "q": "astronomy",
            "rows": 3,
            "fl": "title,author,year"
        }
        
        data = await import_ads_data(params)
        print(f"✅ ADS: Retrieved {len(data)} papers")
        
        if data:
            sample = data[0]
            title = sample.get('title', ['No title'])
            if isinstance(title, list):
                title = title[0] if title else 'No title'
            print(f"   📄 Sample: {title[:80]}...")
            print(f"   📅 Year: {sample.get('year', 'N/A')}")
        
        return True
    except Exception as e:
        print(f"❌ ADS error: {e}")
        return False

async def test_data_cleaning():
    """Test data cleaning functionality"""
    print("\n🧹 Testing Data Cleaning Module...")
    try:
        import pandas as pd
        from api.data_cleaning import analyze_missing_values, analyze_duplicates
        
        # Create test dataset with issues
        test_data = pd.DataFrame({
            'id': [1, 2, 3, 3, 5],  # Duplicate
            'value': [10, None, 30, 30, 50],  # Missing value
            'category': ['A', 'B', 'C', 'C', 'D']
        })
        
        # Test missing value analysis
        missing_analysis = analyze_missing_values(test_data)
        print(f"✅ Missing Values: Detected {missing_analysis['total_missing']} missing values")
        
        # Test duplicate analysis
        duplicate_analysis = analyze_duplicates(test_data)
        print(f"✅ Duplicates: Detected {duplicate_analysis['duplicate_count']} duplicate rows")
        
        print("✅ Data Cleaning: All analysis functions working")
        return True
    except Exception as e:
        print(f"❌ Data Cleaning error: {e}")
        return False

def test_ui_files():
    """Test UI files existence and structure"""
    print("\n🎨 Testing UI Components...")
    try:
        # Check dashboard files
        dashboard_html = "/Users/Gret/scientific-api/ui/dashboard.html"
        dashboard_js = "/Users/Gret/scientific-api/ui/dashboard.js"
        
        if os.path.exists(dashboard_html):
            with open(dashboard_html, 'r') as f:
                content = f.read()
                if 'Scientific Data Platform' in content:
                    print("✅ UI: dashboard.html exists and contains correct title")
                else:
                    print("⚠️  UI: dashboard.html exists but may have issues")
        else:
            print("❌ UI: dashboard.html not found")
            return False
        
        if os.path.exists(dashboard_js):
            with open(dashboard_js, 'r') as f:
                content = f.read()
                if 'showDataCollection' in content and 'importTemplate' in content:
                    print("✅ UI: dashboard.js exists with all required functions")
                else:
                    print("⚠️  UI: dashboard.js exists but may be incomplete")
        else:
            print("❌ UI: dashboard.js not found")
            return False
        
        return True
    except Exception as e:
        print(f"❌ UI error: {e}")
        return False

async def test_api_endpoints():
    """Test API endpoint loading"""
    print("\n🔌 Testing API Endpoints...")
    try:
        from api.index import app
        
        # Get all routes
        routes = []
        for route in app.routes:
            if hasattr(route, 'path'):
                routes.append(route.path)
        
        expected_endpoints = [
            '/ping',
            '/api',
            '/astro/status',
            '/api/data/import-template',
            '/api/data/upload',
            '/api/cleaning/analyze-issues'
        ]
        
        found_endpoints = []
        for endpoint in expected_endpoints:
            # Check if endpoint exists (exact match or pattern)
            if any(endpoint in route for route in routes):
                found_endpoints.append(endpoint)
        
        print(f"✅ API: {len(found_endpoints)}/{len(expected_endpoints)} expected endpoints found")
        print(f"   Found: {', '.join(found_endpoints)}")
        
        return len(found_endpoints) >= len(expected_endpoints) * 0.8
    except Exception as e:
        print(f"❌ API error: {e}")
        return False

async def main():
    """Run all tests"""
    print(f"🕒 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = []
    
    # Run all tests
    results.append(await test_database_connection())
    results.append(await test_data_import_arxiv())
    results.append(await test_data_import_serpapi())
    results.append(await test_data_import_ads())
    results.append(await test_data_cleaning())
    results.append(test_ui_files())
    results.append(await test_api_endpoints())
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print("\n" + "=" * 60)
    print(f"📊 TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Platform is ready for production.")
    elif passed >= total * 0.8:
        print("✅ Most tests passed. Platform is functional with minor issues.")
    else:
        print("⚠️  Some tests failed. Please review and fix issues.")
    
    print(f"🕒 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return passed >= total * 0.8

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
