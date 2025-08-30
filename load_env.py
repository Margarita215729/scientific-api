"""
Load environment variables from config.env file
"""

import os
from dotenv import load_dotenv

# Load environment variables from config.env
config_file = os.path.join(os.path.dirname(__file__), 'config.env')
if os.path.exists(config_file):
    load_dotenv(config_file)
    print(f"Loaded environment variables from {config_file}")
else:
    print(f"Warning: {config_file} not found")

# Verify key variables are loaded
required_vars = ['DB_TYPE', 'MONGODB_URI', 'ADSABS_TOKEN']
for var in required_vars:
    value = os.getenv(var)
    if value:
        print(f"✓ {var}: {'***' + value[-10:] if len(value) > 10 else '***'}")
    else:
        print(f"✗ {var}: Not set")
