#!/usr/bin/env python3
"""
Script to check all environment variables for Scientific API
"""

import os
import sys
from dotenv import load_dotenv

def check_environment_variables():
    """Check all environment variables and their status"""
    
    print("🔍 Checking Scientific API Environment Variables")
    print("=" * 50)
    
    # Load environment variables from env.config
    if os.path.exists('env.config'):
        load_dotenv('env.config')
        print("✅ Loaded variables from env.config")
    else:
        print("❌ env.config file not found")
        return False
    
    # Define all required variables
    required_vars = {
        # Core settings
        'ENVIRONMENT': 'production',
        'PYTHONUNBUFFERED': '1',
        'DEBUG': 'false',
        'LOG_LEVEL': 'INFO',
        
        # SSL settings
        'DISABLE_SSL_VERIFICATION': 'true',
        'PYTHONHTTPSVERIFY': '0',
        
        # API Keys
        'ADSABS_TOKEN': None,
        'SERPAPI_KEY': None,
        'GOOGLE_CLIENT_ID': None,
        'GOOGLE_CLIENT_SECRET': None,
        'GOOGLE_REFRESH_TOKEN': None,
        'HUGGINGFACE_ACCESS_TOKEN': None,
        
        # Database
        'DB_TYPE': 'cosmosdb',
        'AZURE_COSMOS_CONNECTION_STRING': None,
        'COSMOSDB_CONNECTION_STRING': None,
        'MONGODB_URI': None,
        'COSMOS_DB_ACCOUNT': 'scientific-api-server',
        'COSMOS_DB_DATABASE': 'scientific-database',
        'COSMOS_DB_CONTAINER': 'cache',
        'MONGODB_DATABASE_NAME': 'scientific-database',
        'COSMOS_DATABASE_NAME': 'scientific-database',
        'COSMOS_DB_ENDPOINT': None,
        'COSMOS_DB_KEY': None,
        'CACHE_TTL_HOURS': '24',
        'DATABASE_URL': 'sqlite:///./scientific_api.db',
        
        # Security
        'ADMIN_API_KEY': None,
        'USER_API_KEYS': None,
        'RATE_LIMIT_REQUESTS': '100',
        'RATE_LIMIT_WINDOW': '3600',
        
        # Azure settings
        'AZURE_SUBSCRIPTION_ID': None,
        'AZURE_RESOURCE_GROUP': 'scientific-api',
        'AZURE_LOCATION': 'canadacentral',
        'AZURE_APP_NAME': 'scientific-api',
        'AZURE_APP_URL': None,
        
        # App settings
        'WEBSITES_ENABLE_APP_SERVICE_STORAGE': 'false',
        'WEBSITES_PORT': '8000',
        'PORT': '8000',
        'HEAVY_PIPELINE_ON_START': 'false',
        'DEBUG_RELOAD': 'false',
        'WEB_CONCURRENCY': '1'
    }
    
    print("\n📋 Variable Status:")
    print("-" * 50)
    
    all_good = True
    total_vars = len(required_vars)
    working_vars = 0
    
    for var_name, expected_value in required_vars.items():
        actual_value = os.getenv(var_name)
        
        if actual_value is not None:
            working_vars += 1
            if expected_value is None or actual_value == expected_value:
                print(f"✅ {var_name}: {'***' + actual_value[-10:] if len(actual_value) > 10 else actual_value}")
            else:
                print(f"⚠️  {var_name}: {actual_value} (expected: {expected_value})")
                all_good = False
        else:
            print(f"❌ {var_name}: Not set")
            all_good = False
    
    print("\n" + "=" * 50)
    print(f"📊 Summary: {working_vars}/{total_vars} variables working")
    
    if all_good:
        print("🎉 All environment variables are properly configured!")
    else:
        print("⚠️  Some variables need attention")
    
    return all_good

def test_application_load():
    """Test if the application can load with current environment"""
    
    print("\n🧪 Testing Application Load:")
    print("-" * 30)
    
    try:
        # Test basic imports
        from api.config import get_config
        config = get_config()
        print("✅ Configuration module loaded")
        
        # Test database config
        from database.config import DatabaseConfig
        db = DatabaseConfig()
        print(f"✅ Database config: {db.db_type}")
        
        # Test FastAPI app
        from api.index import app
        print("✅ FastAPI application loaded")
        
        print("🎉 Application can start successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Application load failed: {e}")
        return False

if __name__ == "__main__":
    print("Scientific API Environment Checker")
    print("=" * 40)
    
    # Check environment variables
    env_ok = check_environment_variables()
    
    # Test application
    app_ok = test_application_load()
    
    print("\n" + "=" * 40)
    if env_ok and app_ok:
        print("🎉 All checks passed! Application is ready for deployment.")
        sys.exit(0)
    else:
        print("⚠️  Some checks failed. Please review the issues above.")
        sys.exit(1)
