#!/bin/bash

# Script to update existing Azure Container Instance with new image and resources
# Scientific API - Production deployment

set -e

echo "🚀 Updating Scientific API container in Azure..."

# Configuration
RESOURCE_GROUP="scientific-api"
CONTAINER_NAME="scientific-api"
REGISTRY_NAME="scientificapiacr"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}📋 Configuration:${NC}"
echo "  Resource Group: $RESOURCE_GROUP"
echo "  Container Name: $CONTAINER_NAME"
echo "  Registry: $REGISTRY_NAME"
echo ""

# Check if Azure CLI is installed
if ! command -v az &> /dev/null; then
    echo -e "${RED}❌ Azure CLI is not installed. Please install it first.${NC}"
    exit 1
fi

# Login check
echo -e "${YELLOW}🔐 Checking Azure login...${NC}"
if ! az account show &> /dev/null; then
    echo "Please login to Azure:"
    az login
fi

# Get current container info
echo -e "${YELLOW}📊 Getting current container info...${NC}"
CURRENT_STATUS=$(az container show --resource-group $RESOURCE_GROUP --name $CONTAINER_NAME --query "instanceView.state" --output tsv 2>/dev/null || echo "NotFound")

if [ "$CURRENT_STATUS" = "NotFound" ]; then
    echo -e "${RED}❌ Container $CONTAINER_NAME not found in resource group $RESOURCE_GROUP${NC}"
    echo "Available containers:"
    az container list --resource-group $RESOURCE_GROUP --query "[].name" --output table
    exit 1
fi

echo "Current container status: $CURRENT_STATUS"

# Stop the container
echo -e "${YELLOW}⏹️  Stopping current container...${NC}"
az container stop --resource-group $RESOURCE_GROUP --name $CONTAINER_NAME --output table

# Delete the container
echo -e "${YELLOW}🗑️  Deleting current container...${NC}"
az container delete --resource-group $RESOURCE_GROUP --name $CONTAINER_NAME --yes --output table

# Deploy updated container using ARM template
echo -e "${YELLOW}🚀 Deploying updated container...${NC}"
az deployment group create \
    --resource-group $RESOURCE_GROUP \
    --template-file azure-deployment-updated.json \
    --parameters containerGroups_scientific_api_full_name=$CONTAINER_NAME \
    --output table

# Wait for container to be ready
echo -e "${YELLOW}⏳ Waiting for container to start...${NC}"
sleep 30

# Get container details
echo -e "${YELLOW}📊 Getting container details...${NC}"
CONTAINER_IP=$(az container show --resource-group $RESOURCE_GROUP --name $CONTAINER_NAME --query "ipAddress.ip" --output tsv)
CONTAINER_FQDN=$(az container show --resource-group $RESOURCE_GROUP --name $CONTAINER_NAME --query "ipAddress.fqdn" --output tsv)
CONTAINER_STATUS=$(az container show --resource-group $RESOURCE_GROUP --name $CONTAINER_NAME --query "instanceView.state" --output tsv)

echo ""
echo -e "${GREEN}✅ Container update completed!${NC}"
echo ""
echo -e "${GREEN}📊 Container Status: $CONTAINER_STATUS${NC}"
echo -e "${GREEN}🌐 Your Scientific API is available at:${NC}"
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
echo ""
echo -e "${YELLOW}⏳ Note: Container may take 2-3 minutes for full initialization.${NC}"
echo -e "${YELLOW}📊 The API will automatically download and process astronomical catalogs.${NC}"

# Test health endpoint
echo -e "${YELLOW}🔍 Testing health endpoint...${NC}"
sleep 60
curl -f "http://$CONTAINER_FQDN:8000/api/health" || echo -e "${YELLOW}⚠️  Health check failed - container may still be starting${NC}" 