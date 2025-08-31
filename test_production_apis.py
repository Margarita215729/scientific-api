#!/usr/bin/env python3
"""
Production API Testing Script
Tests all major API endpoints with real data to ensure production readiness.
"""

import asyncio
import aiohttp
import json
import sys
import os
from datetime import datetime
from typing import Dict, List, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Test configuration
BASE_URL = "http://localhost:8000"
TEST_TIMEOUT = 30
VERBOSE = True

class APITester:
    """API testing class with comprehensive test coverage."""
    
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.session = None
        self.test_results = []
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=TEST_TIMEOUT))
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def log(self, message: str, level: str = "INFO"):
        """Log test message."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        if VERBOSE or level == "ERROR":
            print(f"[{timestamp}] {level}: {message}")
    
    async def test_endpoint(self, method: str, endpoint: str, data: Dict = None, 
                          expected_status: int = 200, description: str = "") -> Dict[str, Any]:
        """Test a single API endpoint."""
        url = f"{self.base_url}{endpoint}"
        test_name = f"{method} {endpoint}"
        
        try:
            self.log(f"Testing {test_name}: {description}")
            
            if method.upper() == "GET":
                async with self.session.get(url, params=data) as response:
                    status = response.status
                    content = await response.text()
                    
            elif method.upper() == "POST":
                async with self.session.post(url, json=data) as response:
                    status = response.status
                    content = await response.text()
            
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            # Parse JSON response
            try:
                response_data = json.loads(content)
            except json.JSONDecodeError:
                response_data = {"raw_response": content}
            
            # Check status
            success = status == expected_status
            
            result = {
                "test_name": test_name,
                "description": description,
                "status_code": status,
                "expected_status": expected_status,
                "success": success,
                "response_size": len(content),
                "has_data": bool(response_data),
                "timestamp": datetime.now().isoformat()
            }
            
            if success:
                self.log(f"✅ {test_name} - SUCCESS (status: {status})")
            else:
                self.log(f"❌ {test_name} - FAILED (status: {status}, expected: {expected_status})", "ERROR")
                result["error_details"] = response_data
            
            # Add specific checks based on endpoint
            if success and "/api/research" in endpoint:
                result["research_specific"] = self._check_research_response(response_data)
            elif success and "/api/ml" in endpoint:
                result["ml_specific"] = self._check_ml_response(response_data)
            
            self.test_results.append(result)
            return result
            
        except Exception as e:
            self.log(f"❌ {test_name} - EXCEPTION: {str(e)}", "ERROR")
            result = {
                "test_name": test_name,
                "description": description,
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            self.test_results.append(result)
            return result
    
    def _check_research_response(self, data: Dict) -> Dict[str, Any]:
        """Check research API specific response format."""
        checks = {
            "has_papers": "papers" in data or "papers_by_source" in data,
            "has_status": "status" in data,
            "has_timestamp": "timestamp" in data or "search_timestamp" in data,
            "valid_format": isinstance(data, dict)
        }
        return checks
    
    def _check_ml_response(self, data: Dict) -> Dict[str, Any]:
        """Check ML API specific response format."""
        checks = {
            "has_task_info": "task_id" in data or "status" in data,
            "has_message": "message" in data,
            "valid_format": isinstance(data, dict)
        }
        return checks
    
    async def run_comprehensive_tests(self):
        """Run comprehensive API tests."""
        self.log("🚀 Starting comprehensive API tests...")
        
        # Basic health checks
        await self.test_basic_endpoints()
        
        # Research API tests
        await self.test_research_apis()
        
        # ML API tests
        await self.test_ml_apis()
        
        # ADS API tests
        await self.test_ads_apis()
        
        # Data management tests
        await self.test_data_management()
        
        # Generate test report
        self.generate_test_report()
    
    async def test_basic_endpoints(self):
        """Test basic system endpoints."""
        self.log("\n🔍 Testing Basic Endpoints...")
        
        # Health check
        await self.test_endpoint("GET", "/ping", description="Basic health check")
        
        # API info
        await self.test_endpoint("GET", "/api", description="API information")
        
        # Docs endpoint
        await self.test_endpoint("GET", "/docs", expected_status=200, description="API documentation")
    
    async def test_research_apis(self):
        """Test research API endpoints."""
        self.log("\n📚 Testing Research APIs...")
        
        # Research API status
        await self.test_endpoint("GET", "/api/research/status", description="Research API status")
        
        # Unified search
        search_params = {
            "query": "galaxy formation",
            "sources": ["arxiv"],  # Start with ArXiv only for testing
            "max_results_per_source": 5
        }
        await self.test_endpoint("GET", "/api/research/search", data=search_params, 
                                description="Unified research search")
        
        # Astronomy-specific search
        astro_params = {
            "query": "dark matter",
            "max_results": 10,
            "include_preprints": True
        }
        await self.test_endpoint("GET", "/api/research/search/astronomy", data=astro_params,
                                description="Astronomy paper search")
        
        # Recent papers
        recent_params = {
            "days_back": 7,
            "sources": ["arxiv"],
            "category": "astro-ph",
            "limit": 10
        }
        await self.test_endpoint("GET", "/api/research/recent", data=recent_params,
                                description="Recent papers")
        
        # Available categories
        await self.test_endpoint("GET", "/api/research/categories", 
                                description="Available research categories")
    
    async def test_ml_apis(self):
        """Test ML API endpoints."""
        self.log("\n🤖 Testing ML APIs...")
        
        # List models
        await self.test_endpoint("GET", "/api/ml/models", description="List ML models")
        
        # Note: We skip actual ML training test as it takes too long and requires data
        self.log("⏭️  Skipping ML training test (requires authentication and takes time)")
    
    async def test_ads_apis(self):
        """Test ADS API endpoints."""
        self.log("\n🌟 Testing ADS APIs...")
        
        # ADS search by object
        ads_params = {"object_name": "M31"}
        await self.test_endpoint("GET", "/ads/search-by-object", data=ads_params,
                                description="ADS search by object")
        
        # ADS search by coordinates
        coord_params = {"ra": 10.68, "dec": 41.27, "radius": 0.1}
        await self.test_endpoint("GET", "/ads/search-by-coordinates", data=coord_params,
                                description="ADS search by coordinates")
    
    async def test_data_management(self):
        """Test data management endpoints."""
        self.log("\n💾 Testing Data Management...")
        
        # Astro status
        await self.test_endpoint("GET", "/astro/status", description="Astronomical data status")
        
        # Note: We skip data pipeline tests as they require database setup
        self.log("⏭️  Skipping data pipeline tests (require database setup)")
    
    def generate_test_report(self):
        """Generate comprehensive test report."""
        self.log("\n📊 Generating Test Report...")
        
        total_tests = len(self.test_results)
        successful_tests = sum(1 for result in self.test_results if result.get("success", False))
        failed_tests = total_tests - successful_tests
        
        success_rate = (successful_tests / total_tests * 100) if total_tests > 0 else 0
        
        print("\n" + "="*60)
        print("📋 PRODUCTION API TEST REPORT")
        print("="*60)
        print(f"Total Tests: {total_tests}")
        print(f"Successful: {successful_tests} ✅")
        print(f"Failed: {failed_tests} ❌")
        print(f"Success Rate: {success_rate:.1f}%")
        print(f"Test Duration: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if failed_tests > 0:
            print("\n❌ FAILED TESTS:")
            for result in self.test_results:
                if not result.get("success", False):
                    print(f"  - {result['test_name']}: {result.get('error', 'Unknown error')}")
        
        print("\n✅ SUCCESSFUL TESTS:")
        for result in self.test_results:
            if result.get("success", False):
                print(f"  - {result['test_name']}: {result['description']}")
        
        # Save detailed report
        report_file = f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump({
                "summary": {
                    "total_tests": total_tests,
                    "successful_tests": successful_tests,
                    "failed_tests": failed_tests,
                    "success_rate": success_rate,
                    "timestamp": datetime.now().isoformat()
                },
                "detailed_results": self.test_results
            }, f, indent=2)
        
        print(f"\n📄 Detailed report saved to: {report_file}")
        
        # Return overall success
        return success_rate >= 80  # Consider 80%+ success rate as good

async def test_individual_components():
    """Test individual components without full server."""
    print("\n🧪 Testing Individual Components...")
    
    # Test ArXiv API
    try:
        from utils.arxiv_api import arxiv_api
        print("✅ ArXiv API module imported successfully")
        
        # Test search
        result = await arxiv_api.search_papers("galaxy", max_results=2)
        if result.get("papers"):
            print(f"✅ ArXiv search returned {len(result['papers'])} papers")
        else:
            print("⚠️  ArXiv search returned no papers")
    except Exception as e:
        print(f"❌ ArXiv API test failed: {e}")
    
    # Test Semantic Scholar API
    try:
        from utils.semantic_scholar_api import semantic_scholar_api
        print("✅ Semantic Scholar API module imported successfully")
        
        # Test search
        result = await semantic_scholar_api.search_papers("machine learning", max_results=2)
        if result.get("papers"):
            print(f"✅ Semantic Scholar search returned {len(result['papers'])} papers")
        else:
            print("⚠️  Semantic Scholar search returned no papers")
    except Exception as e:
        print(f"❌ Semantic Scholar API test failed: {e}")
    
    # Test ADS API
    try:
        from utils.ads_astronomy_real import ads_client
        print("✅ ADS API module imported successfully")
        
        # Test search
        result = await ads_client.search_publications("galaxy", max_results=2)
        if result.get("publications"):
            print(f"✅ ADS search returned {len(result['publications'])} publications")
        else:
            print("⚠️  ADS search returned no publications")
    except Exception as e:
        print(f"❌ ADS API test failed: {e}")

async def main():
    """Main test function."""
    print("🎯 Scientific API Production Testing Suite")
    print("="*50)
    
    # Test individual components first
    await test_individual_components()
    
    # Test full API if server is running
    try:
        async with APITester() as tester:
            # Test basic connectivity first
            basic_result = await tester.test_endpoint("GET", "/ping", description="Server connectivity")
            
            if basic_result.get("success"):
                print("✅ Server is running, proceeding with full API tests...")
                success = await tester.run_comprehensive_tests()
                
                if success:
                    print("\n🎉 Production API tests completed successfully!")
                    return True
                else:
                    print("\n⚠️  Some tests failed. Check the report for details.")
                    return False
            else:
                print("❌ Server is not responding. Please start the server first:")
                print("   python -m uvicorn api.index:app --reload")
                return False
                
    except Exception as e:
        print(f"❌ Failed to connect to API server: {e}")
        print("💡 Make sure the server is running on http://localhost:8000")
        print("   Start with: python -m uvicorn api.index:app --reload")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
