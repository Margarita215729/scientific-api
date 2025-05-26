#!/bin/bash

# Build lightweight version of Scientific API
# For Vercel proxy or other lightweight deployments
# NOTE: Azure ALWAYS uses full build with database!

set -e

echo "🚀 Building lightweight Scientific API for Vercel/proxy..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="scientific-api"
VERSION="2.0.0"

echo -e "${BLUE}📋 Configuration:${NC}"
echo "  Project: $PROJECT_NAME"
echo "  Version: $VERSION"
echo "  Build Type: Lightweight (Vercel/Proxy only)"
echo "  ${RED}WARNING: Azure ALWAYS uses full build!${NC}"
echo ""

# Build Docker image
echo -e "${YELLOW}🔄 Building lightweight Docker image...${NC}"
docker build --build-arg BUILD_TYPE=lightweight -t $PROJECT_NAME:lightweight-$VERSION .

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Lightweight Docker image built successfully${NC}"
    echo -e "${BLUE}🏷️  Image tag: $PROJECT_NAME:lightweight-$VERSION${NC}"
else
    echo -e "${RED}❌ Docker build failed${NC}"
    exit 1
fi

# Test the container
echo -e "${YELLOW}🔄 Testing lightweight container...${NC}"
docker run --rm -d --name test-lightweight -p 8002:8000 $PROJECT_NAME:lightweight-$VERSION
sleep 5

# Test container health
HEALTH_CHECK=$(curl -s http://localhost:8002/ping || echo "failed")
if [[ $HEALTH_CHECK == *"ok"* ]]; then
    echo -e "${GREEN}✅ Lightweight container test passed${NC}"
    docker stop test-lightweight
else
    echo -e "${RED}❌ Lightweight container test failed${NC}"
    docker stop test-lightweight
    exit 1
fi

echo -e "${GREEN}🎉 Lightweight build completed successfully!${NC}"
echo -e "${BLUE}ℹ️  This build is for Vercel proxy only${NC}"
echo -e "${BLUE}ℹ️  For Azure deployment, use deploy_with_database.sh${NC}" 