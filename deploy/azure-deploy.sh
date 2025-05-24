#!/bin/bash

# Azure Container Instances deployment script for Heavy Compute Service
# Requires Azure CLI to be installed and logged in

set -e

# Configuration
RESOURCE_GROUP="scientific-api"
CONTAINER_NAME="scientific-api-heavy"
IMAGE_NAME="scientific-api-heavy:latest"
REGISTRY_NAME="scientificapiregistry$(date +%s)"
LOCATION="eastus"
SUBSCRIPTION_ID="8e746503-c0c0-4535-a05d-49e544196e3f"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting Azure Container Instances deployment...${NC}"

# Check if Azure CLI is installed
if ! command -v az &> /dev/null; then
    echo -e "${RED}Azure CLI is not installed. Please install it first.${NC}"
    exit 1
fi

# Check if Docker is installed and running
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Docker is not installed. Please install Docker first.${NC}"
    exit 1
fi

if ! docker info &> /dev/null; then
    echo -e "${RED}Docker is not running. Please start Docker first.${NC}"
    exit 1
fi

# Check if logged in to Azure
if ! az account show &> /dev/null; then
    echo -e "${RED}Not logged in to Azure. Please run 'az login' first.${NC}"
    exit 1
fi

# Set the correct subscription
echo -e "${YELLOW}Setting Azure subscription...${NC}"
az account set --subscription $SUBSCRIPTION_ID

# Check if resource group exists, create if it doesn't
echo -e "${YELLOW}Checking resource group...${NC}"
if ! az group show --name $RESOURCE_GROUP &> /dev/null; then
    echo -e "${YELLOW}Creating resource group...${NC}"
    az group create --name $RESOURCE_GROUP --location $LOCATION
else
    echo -e "${GREEN}Resource group $RESOURCE_GROUP already exists${NC}"
fi

# Create Azure Container Registry if it doesn't exist
echo -e "${YELLOW}Creating Azure Container Registry...${NC}"
az acr create --resource-group $RESOURCE_GROUP --name $REGISTRY_NAME --sku Basic --admin-enabled true

# Get ACR login server
ACR_LOGIN_SERVER=$(az acr show --name $REGISTRY_NAME --resource-group $RESOURCE_GROUP --query "loginServer" --output tsv)

# Build and push Docker image
echo -e "${YELLOW}Building Docker image...${NC}"
docker build -f docker/heavy-compute.dockerfile -t $IMAGE_NAME .

# Tag image for ACR
docker tag $IMAGE_NAME $ACR_LOGIN_SERVER/$IMAGE_NAME

# Login to ACR
echo -e "${YELLOW}Logging in to Azure Container Registry...${NC}"
az acr login --name $REGISTRY_NAME

# Push image to ACR
echo -e "${YELLOW}Pushing image to Azure Container Registry...${NC}"
docker push $ACR_LOGIN_SERVER/$IMAGE_NAME

# Get ACR credentials
ACR_USERNAME=$(az acr credential show --name $REGISTRY_NAME --query "username" --output tsv)
ACR_PASSWORD=$(az acr credential show --name $REGISTRY_NAME --query "passwords[0].value" --output tsv)

# Deploy to Azure Container Instances
echo -e "${YELLOW}Deploying to Azure Container Instances...${NC}"
az container create \
    --resource-group $RESOURCE_GROUP \
    --name $CONTAINER_NAME \
    --image $ACR_LOGIN_SERVER/$IMAGE_NAME \
    --registry-login-server $ACR_LOGIN_SERVER \
    --registry-username $ACR_USERNAME \
    --registry-password $ACR_PASSWORD \
    --cpu 12 \
    --memory 20 \
    --dns-name-label scientific-api-heavy \
    --ports 8000 \
    --environment-variables \
        ENVIRONMENT=production \
        ADSABS_TOKEN=$ADSABS_TOKEN \
        SERPAPI_KEY=$SERPAPI_KEY \
        GOOGLE_CLIENT_ID=$GOOGLE_CLIENT_ID \
        GOOGLE_CLIENT_SECRET=$GOOGLE_CLIENT_SECRET \
        GOOGLE_REFRESH_TOKEN=$GOOGLE_REFRESH_TOKEN \
    --restart-policy Always

# Get the FQDN
FQDN=$(az container show --resource-group $RESOURCE_GROUP --name $CONTAINER_NAME --query "ipAddress.fqdn" --output tsv)

echo -e "${GREEN}Deployment completed successfully!${NC}"
echo -e "${GREEN}Heavy Compute Service URL: http://$FQDN:8000${NC}"
echo -e "${GREEN}Health check: http://$FQDN:8000/ping${NC}"

# Test the deployment
echo -e "${YELLOW}Testing deployment...${NC}"
sleep 30  # Wait for container to start

if curl -f "http://$FQDN:8000/ping" > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Heavy Compute Service is running successfully!${NC}"
else
    echo -e "${RED}✗ Heavy Compute Service health check failed${NC}"
    echo -e "${YELLOW}Check container logs with: az container logs --resource-group $RESOURCE_GROUP --name $CONTAINER_NAME${NC}"
fi

echo -e "${GREEN}Deployment script completed.${NC}"
echo -e "${YELLOW}Don't forget to update your Vercel environment variables:${NC}"
echo -e "${YELLOW}HEAVY_COMPUTE_URL=http://$FQDN:8000${NC}" 