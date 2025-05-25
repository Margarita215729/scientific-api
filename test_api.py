#!/usr/bin/env python3

import sys
import traceback
sys.path.append('.')

try:
    from api.heavy_api import get_catalog_data
    print("✅ Successfully imported get_catalog_data")
    
    # Test the function
    result = get_catalog_data(limit=2)
    print(f"✅ Function works: {len(result)} objects returned")
    
    # Test with filters
    result_filtered = get_catalog_data(limit=1, min_z=0.1, max_z=0.5)
    print(f"✅ Filtered function works: {len(result_filtered)} objects returned")
    
except Exception as e:
    print(f"❌ Error: {e}")
    traceback.print_exc()

try:
    from fastapi.testclient import TestClient
    from api.heavy_api import app
    
    client = TestClient(app)
    
    print("\n🧪 Testing API endpoints...")
    
    # Test ping
    response = client.get("/ping")
    print(f"Ping status: {response.status_code}")
    if response.status_code != 200:
        print(f"Ping error: {response.text}")
    
    # Test galaxies endpoint
    response = client.get("/astro/galaxies?limit=2")
    print(f"Galaxies status: {response.status_code}")
    if response.status_code != 200:
        print(f"Galaxies error: {response.text}")
    else:
        data = response.json()
        print(f"✅ Galaxies endpoint works: {data.get('count', 0)} objects")
        
except Exception as e:
    print(f"❌ API Test Error: {e}")
    traceback.print_exc() 