#!/bin/bash

# Production deployment script for Azure Container Instances
# Scientific API - Full astronomical data processing

set -e

echo "🚀 Starting production deployment to Azure Container Instances..."

# Configuration
RESOURCE_GROUP="scientific-api-production"
CONTAINER_NAME="scientific-api-prod"
IMAGE_NAME="scientific-api:production"
REGISTRY_NAME="scientificapiregistry"
LOCATION="eastus"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}📋 Configuration:${NC}"
echo "  Resource Group: $RESOURCE_GROUP"
echo "  Container Name: $CONTAINER_NAME"
echo "  Image: $IMAGE_NAME"
echo "  Registry: $REGISTRY_NAME"
echo "  Location: $LOCATION"
echo ""

# Check if Azure CLI is installed
if ! command -v az &> /dev/null; then
    echo -e "${RED}❌ Azure CLI is not installed. Please install it first.${NC}"
    echo "Visit: https://docs.microsoft.com/en-us/cli/azure/install-azure-cli"
    exit 1
fi

# Login to Azure (if not already logged in)
echo -e "${YELLOW}🔐 Checking Azure login...${NC}"
if ! az account show &> /dev/null; then
    echo "Please login to Azure:"
    az login
fi

# Create resource group if it doesn't exist
echo -e "${YELLOW}📦 Creating resource group...${NC}"
az group create --name $RESOURCE_GROUP --location $LOCATION --output table

# Create Azure Container Registry if it doesn't exist
echo -e "${YELLOW}🏗️  Creating Azure Container Registry...${NC}"
az acr create \
    --resource-group $RESOURCE_GROUP \
    --name $REGISTRY_NAME \
    --sku Basic \
    --admin-enabled true \
    --output table

# Get ACR login server
ACR_LOGIN_SERVER=$(az acr show --name $REGISTRY_NAME --resource-group $RESOURCE_GROUP --query "loginServer" --output tsv)
echo "Registry login server: $ACR_LOGIN_SERVER"

# Build and push Docker image
echo -e "${YELLOW}🐳 Building Docker image...${NC}"
docker build -t $IMAGE_NAME .

echo -e "${YELLOW}🏷️  Tagging image for ACR...${NC}"
docker tag $IMAGE_NAME $ACR_LOGIN_SERVER/$IMAGE_NAME

echo -e "${YELLOW}🔐 Logging into ACR...${NC}"
az acr login --name $REGISTRY_NAME

echo -e "${YELLOW}📤 Pushing image to ACR...${NC}"
docker push $ACR_LOGIN_SERVER/$IMAGE_NAME

# Get ACR credentials
echo -e "${YELLOW}🔑 Getting ACR credentials...${NC}"
ACR_USERNAME=$(az acr credential show --name $REGISTRY_NAME --resource-group $RESOURCE_GROUP --query "username" --output tsv)
ACR_PASSWORD=$(az acr credential show --name $REGISTRY_NAME --resource-group $RESOURCE_GROUP --query "passwords[0].value" --output tsv)

# Delete existing container if it exists
echo -e "${YELLOW}🗑️  Removing existing container (if any)...${NC}"
az container delete \
    --resource-group $RESOURCE_GROUP \
    --name $CONTAINER_NAME \
    --yes \
    --output table 2>/dev/null || true

# Deploy to Azure Container Instances
echo -e "${YELLOW}🚀 Deploying to Azure Container Instances...${NC}"
az container create \
    --resource-group $RESOURCE_GROUP \
    --name $CONTAINER_NAME \
    --image $ACR_LOGIN_SERVER/$IMAGE_NAME \
    --registry-login-server $ACR_LOGIN_SERVER \
    --registry-username $ACR_USERNAME \
    --registry-password $ACR_PASSWORD \
    --dns-name-label $CONTAINER_NAME \
    --ports 8000 \
    --cpu 4 \
    --memory 8 \
    --environment-variables \
        ENVIRONMENT=production \
        PYTHONPATH=/app \
        PYTHONUNBUFFERED=1 \
    --restart-policy Always \
    --output table

# Get container details
echo -e "${YELLOW}📊 Getting container details...${NC}"
CONTAINER_IP=$(az container show --resource-group $RESOURCE_GROUP --name $CONTAINER_NAME --query "ipAddress.ip" --output tsv)
CONTAINER_FQDN=$(az container show --resource-group $RESOURCE_GROUP --name $CONTAINER_NAME --query "ipAddress.fqdn" --output tsv)

echo ""
echo -e "${GREEN}✅ Deployment completed successfully!${NC}"
echo ""
echo -e "${GREEN}🌐 Your Scientific API is now available at:${NC}"
echo "  Public IP: http://$CONTAINER_IP:8000"
echo "  FQDN: http://$CONTAINER_FQDN:8000"
echo ""
echo -e "${GREEN}📚 API Endpoints:${NC}"
echo "  Health Check: http://$CONTAINER_FQDN:8000/api/health"
echo "  API Docs: http://$CONTAINER_FQDN:8000/api/docs"
echo "  Web Interface: http://$CONTAINER_FQDN:8000/"
echo "  ADS Search: http://$CONTAINER_FQDN:8000/ads"
echo ""
echo -e "${GREEN}🔧 Management Commands:${NC}"
echo "  View logs: az container logs --resource-group $RESOURCE_GROUP --name $CONTAINER_NAME"
echo "  Restart: az container restart --resource-group $RESOURCE_GROUP --name $CONTAINER_NAME"
echo "  Delete: az container delete --resource-group $RESOURCE_GROUP --name $CONTAINER_NAME --yes"
echo ""
echo -e "${YELLOW}⏳ Note: Container startup may take 2-3 minutes for full initialization.${NC}"
echo -e "${YELLOW}📊 The API will automatically download and process astronomical catalogs on first run.${NC}" 