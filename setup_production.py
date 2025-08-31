#!/usr/bin/env python3
"""
Production setup and validation script for Scientific API.
This script installs dependencies, validates configuration, and tests all APIs.
"""

import subprocess
import sys
import os
import asyncio
from pathlib import Path
import json
from typing import Dict, List, Any

def run_command(cmd: List[str], description: str) -> bool:
    """Run a command and return success status."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ {description} - SUCCESS")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - FAILED")
        print(f"Error: {e.stderr}")
        return False

def check_python_version() -> bool:
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python version {version.major}.{version.minor}.{version.micro} is compatible")
        return True
    else:
        print(f"❌ Python version {version.major}.{version.minor}.{version.micro} is not compatible (requires 3.8+)")
        return False

def install_dependencies() -> bool:
    """Install all required dependencies."""
    print("\n📦 Installing dependencies...")
    
    # Upgrade pip first
    if not run_command([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], 
                      "Upgrading pip"):
        return False
    
    # Install requirements
    if not run_command([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      "Installing requirements"):
        return False
    
    return True

def validate_environment() -> Dict[str, Any]:
    """Validate environment configuration."""
    print("\n🔍 Validating environment configuration...")
    
    validation_results = {
        "api_keys": {},
        "database": {},
        "ssl_config": {},
        "dependencies": {}
    }
    
    # Check API keys
    api_keys = {
        "ADSABS_TOKEN": "NASA ADS API",
        "SERPAPI_KEY": "SerpAPI",
        "GOOGLE_CLIENT_ID": "Google APIs (optional)",
        "ADMIN_API_KEY": "Admin authentication"
    }
    
    for key, description in api_keys.items():
        value = os.getenv(key)
        validation_results["api_keys"][key] = {
            "configured": bool(value),
            "description": description,
            "required": key in ["ADSABS_TOKEN", "ADMIN_API_KEY"]
        }
        
        if value:
            print(f"✅ {description} - configured")
        else:
            required = key in ["ADSABS_TOKEN", "ADMIN_API_KEY"]
            symbol = "❌" if required else "⚠️"
            status = "REQUIRED" if required else "optional"
            print(f"{symbol} {description} - not configured ({status})")
    
    # Check database configuration
    db_type = os.getenv("DB_TYPE", "sqlite")
    validation_results["database"]["type"] = db_type
    
    if db_type == "cosmosdb":
        cosmos_conn = os.getenv("AZURE_COSMOS_CONNECTION_STRING")
        validation_results["database"]["cosmosdb_configured"] = bool(cosmos_conn)
        if cosmos_conn:
            print("✅ Cosmos DB connection string - configured")
        else:
            print("❌ Cosmos DB connection string - not configured")
    
    # Check SSL configuration
    ssl_verify = os.getenv("PYTHONHTTPSVERIFY", "1")
    validation_results["ssl_config"]["verify_enabled"] = ssl_verify == "1"
    print(f"✅ SSL verification - {'enabled' if ssl_verify == '1' else 'disabled'}")
    
    return validation_results

async def test_api_integrations() -> Dict[str, Any]:
    """Test all API integrations."""
    print("\n🧪 Testing API integrations...")
    
    test_results = {}
    
    try:
        # Test ArXiv API
        print("Testing ArXiv API...")
        sys.path.insert(0, '.')
        from utils.arxiv_api import arxiv_api
        
        result = await arxiv_api.search_papers("galaxy", max_results=1)
        test_results["arxiv"] = {
            "status": result.get("status"),
            "working": result.get("status") not in ["error"],
            "papers_found": len(result.get("papers", []))
        }
        
        if test_results["arxiv"]["working"]:
            print("✅ ArXiv API - working")
        else:
            print(f"❌ ArXiv API - {result.get('status', 'error')}")
    
    except Exception as e:
        test_results["arxiv"] = {"status": "error", "working": False, "error": str(e)}
        print(f"❌ ArXiv API - exception: {e}")
    
    try:
        # Test Semantic Scholar API
        print("Testing Semantic Scholar API...")
        from utils.semantic_scholar_api import semantic_scholar_api
        
        result = await semantic_scholar_api.search_papers("machine learning", max_results=1)
        test_results["semantic_scholar"] = {
            "status": result.get("status"),
            "working": result.get("status") == "success",
            "papers_found": len(result.get("papers", []))
        }
        
        if test_results["semantic_scholar"]["working"]:
            print("✅ Semantic Scholar API - working")
        else:
            print(f"❌ Semantic Scholar API - {result.get('status', 'error')}")
    
    except Exception as e:
        test_results["semantic_scholar"] = {"status": "error", "working": False, "error": str(e)}
        print(f"❌ Semantic Scholar API - exception: {e}")
    
    try:
        # Test ADS API
        print("Testing ADS API...")
        from utils.ads_astronomy_real import ads_client
        
        result = await ads_client.search_publications("galaxy", max_results=1)
        test_results["ads"] = {
            "status": result.get("status"),
            "working": len(result.get("publications", [])) > 0,
            "papers_found": len(result.get("publications", []))
        }
        
        if test_results["ads"]["working"]:
            print("✅ ADS API - working")
        else:
            print(f"❌ ADS API - {result.get('status', 'no results')}")
    
    except Exception as e:
        test_results["ads"] = {"status": "error", "working": False, "error": str(e)}
        print(f"❌ ADS API - exception: {e}")
    
    return test_results

def test_dependencies() -> Dict[str, bool]:
    """Test if all required dependencies are available."""
    print("\n🔧 Testing dependencies...")
    
    dependencies = {
        "pandas": "Data processing",
        "numpy": "Scientific computing",
        "sklearn": "Machine learning",
        "astropy": "Astronomy",
        "motor": "MongoDB driver",
        "asyncpg": "PostgreSQL driver",
        "requests": "HTTP client",
        "feedparser": "RSS parsing",
        "certifi": "SSL certificates"
    }
    
    results = {}
    
    for package, description in dependencies.items():
        try:
            if package == "sklearn":
                import sklearn
            else:
                __import__(package)
            results[package] = True
            print(f"✅ {description} ({package}) - available")
        except ImportError:
            results[package] = False
            print(f"❌ {description} ({package}) - not available")
    
    return results

def generate_report(validation_results: Dict, test_results: Dict, dependency_results: Dict):
    """Generate a comprehensive setup report."""
    print("\n📊 PRODUCTION READINESS REPORT")
    print("=" * 50)
    
    # Calculate scores
    api_keys_score = sum(1 for k, v in validation_results["api_keys"].items() 
                        if v["configured"] or not v["required"]) / len(validation_results["api_keys"])
    
    api_tests_score = sum(1 for result in test_results.values() 
                         if result.get("working", False)) / len(test_results) if test_results else 0
    
    deps_score = sum(dependency_results.values()) / len(dependency_results)
    
    overall_score = (api_keys_score + api_tests_score + deps_score) / 3
    
    print(f"Overall Readiness Score: {overall_score:.1%}")
    print(f"API Keys Configuration: {api_keys_score:.1%}")
    print(f"API Integration Tests: {api_tests_score:.1%}")
    print(f"Dependencies: {deps_score:.1%}")
    
    print("\n🚨 ISSUES TO FIX:")
    
    # Missing API keys
    missing_keys = [k for k, v in validation_results["api_keys"].items() 
                   if not v["configured"] and v["required"]]
    if missing_keys:
        print("Missing required API keys:")
        for key in missing_keys:
            print(f"  - {key}")
    
    # Failed API tests
    failed_apis = [api for api, result in test_results.items() 
                  if not result.get("working", False)]
    if failed_apis:
        print("Failed API integrations:")
        for api in failed_apis:
            print(f"  - {api}")
    
    # Missing dependencies
    missing_deps = [dep for dep, available in dependency_results.items() if not available]
    if missing_deps:
        print("Missing dependencies:")
        for dep in missing_deps:
            print(f"  - {dep}")
    
    if overall_score >= 0.8:
        print("\n🎉 READY FOR PRODUCTION!")
    elif overall_score >= 0.6:
        print("\n⚠️  PARTIALLY READY - Fix issues above")
    else:
        print("\n❌ NOT READY FOR PRODUCTION - Major issues need fixing")
    
    # Save detailed report
    report_data = {
        "timestamp": asyncio.get_event_loop().time(),
        "overall_score": overall_score,
        "validation_results": validation_results,
        "test_results": test_results,
        "dependency_results": dependency_results
    }
    
    with open("production_readiness_report.json", "w") as f:
        json.dump(report_data, f, indent=2)
    
    print(f"\n📄 Detailed report saved to: production_readiness_report.json")

async def main():
    """Main setup and validation process."""
    print("🚀 Scientific API Production Setup & Validation")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Failed to install dependencies")
        sys.exit(1)
    
    # Validate environment
    validation_results = validate_environment()
    
    # Test dependencies
    dependency_results = test_dependencies()
    
    # Test API integrations
    test_results = await test_api_integrations()
    
    # Generate report
    generate_report(validation_results, test_results, dependency_results)

if __name__ == "__main__":
    asyncio.run(main())

