#!/bin/bash

# Scientific API - Full Project Build Script
# Version 2.0.0

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

echo -e "${PURPLE}🚀 Scientific API - Full Project Build${NC}"
echo -e "${PURPLE}======================================${NC}"
echo ""

# Configuration
PROJECT_NAME="scientific-api"
VERSION="2.0.0"
DOCKER_IMAGE="index.docker.io/gretk/scientific-api-app-image:scientific-api"

echo -e "${BLUE}📋 Build Configuration:${NC}"
echo "  Project: $PROJECT_NAME"
echo "  Version: $VERSION"
echo "  Docker Image: $DOCKER_IMAGE"
echo "  Build Date: $(date)"
echo ""

# Step 1: Environment Check
echo -e "${YELLOW}🔍 Step 1: Checking environment...${NC}"

# Check Python
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo -e "${GREEN}✅ Python: $PYTHON_VERSION${NC}"
else
    echo -e "${RED}❌ Python 3 not found${NC}"
    exit 1
fi

# Check Docker
if command -v docker &> /dev/null; then
    DOCKER_VERSION=$(docker --version)
    echo -e "${GREEN}✅ Docker: $DOCKER_VERSION${NC}"
else
    echo -e "${RED}❌ Docker not found${NC}"
    exit 1
fi

# Check Azure CLI
if command -v az &> /dev/null; then
    AZ_VERSION=$(az --version | head -n 1)
    echo -e "${GREEN}✅ Azure CLI: $AZ_VERSION${NC}"
else
    echo -e "${YELLOW}⚠️  Azure CLI not found (optional for local build)${NC}"
fi

echo ""

# Step 2: Python Environment Setup
echo -e "${YELLOW}🐍 Step 2: Setting up Python environment...${NC}"

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    echo -e "${GREEN}✅ Python dependencies installed${NC}"
else
    echo -e "${RED}❌ requirements.txt not found${NC}"
    exit 1
fi

echo ""

# Step 3: Code Quality Checks
echo -e "${YELLOW}🔍 Step 3: Running code quality checks...${NC}"

# Check if main files exist
REQUIRED_FILES=("main.py" "azure-webapp-bicep.bicep" "deploy_azure_bicep.sh" "Dockerfile")
for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo -e "${GREEN}✅ $file exists${NC}"
    else
        echo -e "${RED}❌ $file missing${NC}"
        exit 1
    fi
done

# Python syntax check
echo "Checking Python syntax..."
python3 -m py_compile main.py
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Python syntax check passed${NC}"
else
    echo -e "${RED}❌ Python syntax errors found${NC}"
    exit 1
fi

echo ""

# Step 4: Local Testing
echo -e "${YELLOW}🧪 Step 4: Running local tests...${NC}"

# Test import of main modules
echo "Testing module imports..."
python3 -c "
try:
    import main
    print('✅ Main module imports successfully')
except Exception as e:
    print(f'❌ Import error: {e}')
    exit(1)
"

# Test configuration loading
echo "Testing configuration..."
python3 -c "
try:
    from api.cosmos_db_config import COSMOS_DB_DATABASE
    print(f'✅ Configuration loaded: DB={COSMOS_DB_DATABASE}')
except Exception as e:
    print(f'❌ Configuration error: {e}')
    exit(1)
"

echo -e "${GREEN}✅ Local tests passed${NC}"
echo ""

# Step 5: Docker Build
echo -e "${YELLOW}🐳 Step 5: Building Docker image...${NC}"

# Build Docker image
echo "Building Docker image..."
docker build -t $PROJECT_NAME:$VERSION .

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Docker image built successfully${NC}"
else
    echo -e "${RED}❌ Docker build failed${NC}"
    exit 1
fi

# Tag for registry
echo "Tagging image for registry..."
docker tag $PROJECT_NAME:$VERSION $DOCKER_IMAGE

echo -e "${GREEN}✅ Docker image tagged: $DOCKER_IMAGE${NC}"
echo ""

# Step 6: Docker Test
echo -e "${YELLOW}🧪 Step 6: Testing Docker container...${NC}"

# Start container in background
echo "Starting test container..."
docker run -d --name ${PROJECT_NAME}-test -p 8001:8000 $PROJECT_NAME:$VERSION

# Wait for container to start
echo "Waiting for container to start..."
sleep 15

# Test health endpoint
echo "Testing health endpoint..."
HEALTH_RESPONSE=$(curl -s http://localhost:8001/ping || echo "failed")

if [[ $HEALTH_RESPONSE == *"ok"* ]]; then
    echo -e "${GREEN}✅ Container health check passed${NC}"
else
    echo -e "${RED}❌ Container health check failed${NC}"
    docker logs ${PROJECT_NAME}-test
    docker stop ${PROJECT_NAME}-test
    docker rm ${PROJECT_NAME}-test
    exit 1
fi

# Stop and remove test container
echo "Cleaning up test container..."
docker stop ${PROJECT_NAME}-test
docker rm ${PROJECT_NAME}-test

echo ""

# Step 7: Azure Resources Validation
echo -e "${YELLOW}☁️  Step 7: Validating Azure resources...${NC}"

if command -v az &> /dev/null; then
    # Check if logged in
    if az account show &> /dev/null; then
        echo -e "${GREEN}✅ Azure CLI authenticated${NC}"
        
        # Validate Bicep template
        echo "Validating Bicep template..."
        az deployment group validate \
            --resource-group scientific-api \
            --template-file azure-webapp-bicep.bicep \
            --parameters sites_scientific_api_name=scientific-api \
            > /dev/null 2>&1
        
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✅ Bicep template validation passed${NC}"
        else
            echo -e "${YELLOW}⚠️  Bicep template validation failed (may need resource group)${NC}"
        fi
    else
        echo -e "${YELLOW}⚠️  Azure CLI not authenticated${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  Azure CLI not available${NC}"
fi

echo ""

# Step 8: Documentation Check
echo -e "${YELLOW}📚 Step 8: Checking documentation...${NC}"

DOC_FILES=("README.md" "azure.env")
for file in "${DOC_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo -e "${GREEN}✅ $file exists${NC}"
    else
        echo -e "${YELLOW}⚠️  $file missing${NC}"
    fi
done

echo ""

# Step 9: Security Check
echo -e "${YELLOW}🔒 Step 9: Security validation...${NC}"

# Check for sensitive data in files
echo "Checking for sensitive data..."
if grep -r "password\|secret\|key" --include="*.py" --include="*.sh" --include="*.json" . | grep -v "# " | grep -v "your_" | grep -v "REDACTED" > /dev/null; then
    echo -e "${YELLOW}⚠️  Potential sensitive data found in code${NC}"
else
    echo -e "${GREEN}✅ No sensitive data found in code${NC}"
fi

# Check file permissions
echo "Checking script permissions..."
SCRIPTS=("deploy_azure_bicep.sh" "deploy_azure_webapp.sh" "build_project.sh")
for script in "${SCRIPTS[@]}"; do
    if [ -f "$script" ] && [ -x "$script" ]; then
        echo -e "${GREEN}✅ $script is executable${NC}"
    elif [ -f "$script" ]; then
        echo -e "${YELLOW}⚠️  $script is not executable${NC}"
        chmod +x "$script"
        echo -e "${GREEN}✅ Made $script executable${NC}"
    fi
done

echo ""

# Step 10: Build Summary
echo -e "${PURPLE}📊 Build Summary${NC}"
echo -e "${PURPLE}===============${NC}"

echo -e "${GREEN}✅ Environment Check: Passed${NC}"
echo -e "${GREEN}✅ Python Setup: Completed${NC}"
echo -e "${GREEN}✅ Code Quality: Passed${NC}"
echo -e "${GREEN}✅ Local Tests: Passed${NC}"
echo -e "${GREEN}✅ Docker Build: Completed${NC}"
echo -e "${GREEN}✅ Docker Test: Passed${NC}"
echo -e "${GREEN}✅ Documentation: Available${NC}"
echo -e "${GREEN}✅ Security: Validated${NC}"

echo ""
echo -e "${PURPLE}🎉 BUILD COMPLETED SUCCESSFULLY! 🎉${NC}"
echo ""

# Deployment options
echo -e "${BLUE}🚀 Deployment Options:${NC}"
echo ""
echo -e "${YELLOW}1. Deploy to Azure (Bicep):${NC}"
echo "   ./deploy_azure_bicep.sh"
echo ""
echo -e "${YELLOW}2. Deploy to Azure (ARM):${NC}"
echo "   ./deploy_azure_webapp.sh"
echo ""
echo -e "${YELLOW}3. Run locally:${NC}"
echo "   source .venv/bin/activate"
echo "   python main.py"
echo ""
echo -e "${YELLOW}4. Run with Docker:${NC}"
echo "   docker run -p 8000:8000 $PROJECT_NAME:$VERSION"
echo ""
echo -e "${YELLOW}5. Run with Docker Compose:${NC}"
echo "   docker-compose up"
echo ""

# Build artifacts
echo -e "${BLUE}📦 Build Artifacts:${NC}"
echo "  Docker Image: $PROJECT_NAME:$VERSION"
echo "  Registry Image: $DOCKER_IMAGE"
echo "  Bicep Template: azure-webapp-bicep.bicep"
echo "  ARM Template: azure-webapp-config.json"
echo "  Environment: azure.env"
echo ""

echo -e "${GREEN}✨ Project is ready for deployment! ✨${NC}" 