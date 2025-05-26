#!/bin/bash

# Scientific API Deployment Script with Database
# Version 2.0.0

set -e

echo "🚀 Starting Scientific API deployment with database..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="scientific-api"
VERSION="2.0.0"
AZURE_RESOURCE_GROUP="scientific-api-rg"
AZURE_LOCATION="eastus"

echo -e "${BLUE}📋 Configuration:${NC}"
echo "  Project: $PROJECT_NAME"
echo "  Version: $VERSION"
echo "  Resource Group: $AZURE_RESOURCE_GROUP"
echo "  Location: $AZURE_LOCATION"
echo ""

# Step 1: Initialize database locally
echo -e "${YELLOW}🔄 Step 1: Initializing database locally...${NC}"
python3 init_database.py
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Database initialized successfully${NC}"
else
    echo -e "${RED}❌ Database initialization failed${NC}"
    exit 1
fi

# Step 2: Test the application locally
echo -e "${YELLOW}🔄 Step 2: Testing application locally...${NC}"
python3 -c "
import asyncio
from main_azure_with_db import app
from database.config import db

async def test_app():
    try:
        # Test database connection
        stats = await db.get_statistics()
        print(f'Database stats: {len(stats)} metrics')
        
        objects = await db.get_astronomical_objects(limit=3)
        print(f'Sample objects: {len(objects)} loaded')
        
        print('✅ Application test passed')
        return True
    except Exception as e:
        print(f'❌ Application test failed: {e}')
        return False
    finally:
        await db.disconnect()

result = asyncio.run(test_app())
exit(0 if result else 1)
"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Application test passed${NC}"
else
    echo -e "${RED}❌ Application test failed${NC}"
    exit 1
fi

# Step 3: Build Docker image
echo -e "${YELLOW}🔄 Step 3: Building Docker image...${NC}"
docker build -f Dockerfile.azure.db -t $PROJECT_NAME:db-$VERSION .
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Docker image built successfully${NC}"
else
    echo -e "${RED}❌ Docker build failed${NC}"
    exit 1
fi

# Step 4: Test Docker container locally
echo -e "${YELLOW}🔄 Step 4: Testing Docker container locally...${NC}"
docker run --rm -d --name test-container -p 8001:8000 $PROJECT_NAME:db-$VERSION
sleep 10

# Test container health
HEALTH_CHECK=$(curl -s http://localhost:8001/ping || echo "failed")
if [[ $HEALTH_CHECK == *"ok"* ]]; then
    echo -e "${GREEN}✅ Docker container test passed${NC}"
    docker stop test-container
else
    echo -e "${RED}❌ Docker container test failed${NC}"
    docker stop test-container
    exit 1
fi

# Step 5: Deploy to Azure (optional)
read -p "🤔 Deploy to Azure? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}🔄 Step 5: Deploying to Azure...${NC}"
    
    # Check if Azure CLI is installed
    if ! command -v az &> /dev/null; then
        echo -e "${RED}❌ Azure CLI not found. Please install it first.${NC}"
        exit 1
    fi
    
    # Login check
    az account show &> /dev/null || {
        echo -e "${YELLOW}🔐 Please login to Azure...${NC}"
        az login
    }
    
    # Create resource group if it doesn't exist
    az group create --name $AZURE_RESOURCE_GROUP --location $AZURE_LOCATION
    
    # Deploy container instance
    az deployment group create \
        --resource-group $AZURE_RESOURCE_GROUP \
        --template-file azure-deployment-with-db.json \
        --parameters containerGroupName=$PROJECT_NAME-db-$(date +%s) \
                    image=$PROJECT_NAME:db-$VERSION
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Azure deployment completed${NC}"
        
        # Get deployment info
        FQDN=$(az container show --resource-group $AZURE_RESOURCE_GROUP --name $PROJECT_NAME-db-* --query ipAddress.fqdn --output tsv 2>/dev/null || echo "Not available")
        IP=$(az container show --resource-group $AZURE_RESOURCE_GROUP --name $PROJECT_NAME-db-* --query ipAddress.ip --output tsv 2>/dev/null || echo "Not available")
        
        echo -e "${BLUE}🌐 Deployment Information:${NC}"
        echo "  FQDN: $FQDN"
        echo "  IP: $IP"
        echo "  URL: http://$FQDN:8000"
    else
        echo -e "${RED}❌ Azure deployment failed${NC}"
        exit 1
    fi
else
    echo -e "${BLUE}ℹ️  Skipping Azure deployment${NC}"
fi

# Step 6: Deploy to Vercel (optional)
read -p "🤔 Deploy frontend to Vercel? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}🔄 Step 6: Deploying to Vercel...${NC}"
    
    # Check if Vercel CLI is installed
    if ! command -v vercel &> /dev/null; then
        echo -e "${YELLOW}📦 Installing Vercel CLI...${NC}"
        npm install -g vercel
    fi
    
    # Copy clean files for Vercel
    cp main_vercel.py main.py
    cp requirements_vercel.txt requirements.txt
    cp vercel.json vercel.json
    
    # Deploy to Vercel
    vercel --prod
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Vercel deployment completed${NC}"
    else
        echo -e "${RED}❌ Vercel deployment failed${NC}"
    fi
    
    # Cleanup
    rm -f main.py requirements.txt vercel.json
else
    echo -e "${BLUE}ℹ️  Skipping Vercel deployment${NC}"
fi
