#!/bin/bash

# Azure Container Deployment Script
# This script builds and deploys the scientific API to Azure Container Instances

set -e

# Configuration
RESOURCE_GROUP="scientific-api"
CONTAINER_NAME="scientific-api"
REGISTRY_NAME="scientificapiacr"
IMAGE_NAME="gretk/scientific-api-app-image"
LOCATION="canadacentral"

echo "============================================"
echo "Azure Container Deployment for Scientific API"
echo "============================================"

# Check if Azure CLI is installed
if ! command -v az &> /dev/null; then
    echo "❌ Azure CLI is not installed. Please install it first."
    exit 1
fi

# Login to Azure (if not already logged in)
echo "🔐 Checking Azure login status..."
if ! az account show &> /dev/null; then
    echo "🔐 Please login to Azure..."
    az login
fi

# Build and push Docker image
echo "🔨 Building Docker image..."
docker build -t ${IMAGE_NAME}:latest .

echo "🏷️  Tagging image for Azure Container Registry..."
docker tag ${IMAGE_NAME}:latest ${REGISTRY_NAME}.azurecr.io/${IMAGE_NAME}:latest

echo "🔐 Logging into Azure Container Registry..."
az acr login --name ${REGISTRY_NAME}

echo "📤 Pushing image to Azure Container Registry..."
docker push ${REGISTRY_NAME}.azurecr.io/${IMAGE_NAME}:latest

# Get ACR credentials
echo "🔑 Getting ACR credentials..."
ACR_SERVER=$(az acr show --name ${REGISTRY_NAME} --query loginServer --output tsv)
ACR_USERNAME=$(az acr credential show --name ${REGISTRY_NAME} --query username --output tsv)
ACR_PASSWORD=$(az acr credential show --name ${REGISTRY_NAME} --query passwords[0].value --output tsv)

# Deploy to Azure Container Instances
echo "🚀 Deploying to Azure Container Instances..."
az container create \
    --resource-group ${RESOURCE_GROUP} \
    --name ${CONTAINER_NAME} \
    --image ${ACR_SERVER}/${IMAGE_NAME}:latest \
    --cpu 2 \
    --memory 4 \
    --registry-login-server ${ACR_SERVER} \
    --registry-username ${ACR_USERNAME} \
    --registry-password ${ACR_PASSWORD} \
    --dns-name-label ${CONTAINER_NAME}-$(date +%s) \
    --ports 8000 \
    --environment-variables \
        ENVIRONMENT=production \
        PYTHONUNBUFFERED=1 \
        ADSABS_TOKEN="pDbJnrgwpsZYj14Vbkn1LmMCjdiIxXrgUcrdLvjk" \
        SERPAPI_KEY="12ad8fb8fbcd01088827b7d3143de6d9fa15c571dc11ac0daf7f24b25097fac8" \
    --restart-policy Always

# Get the FQDN
echo "📋 Getting container information..."
FQDN=$(az container show --resource-group ${RESOURCE_GROUP} --name ${CONTAINER_NAME} --query ipAddress.fqdn --output tsv)
IP=$(az container show --resource-group ${RESOURCE_GROUP} --name ${CONTAINER_NAME} --query ipAddress.ip --output tsv)

echo ""
echo "✅ Deployment completed successfully!"
echo "============================================"
echo "📡 Container FQDN: ${FQDN}"
echo "🌐 IP Address: ${IP}"
echo "🔗 API URL: http://${FQDN}:8000"
echo "🏥 Health Check: http://${FQDN}:8000/ping"
echo "📊 Status: http://${FQDN}:8000/astro/status"
echo "============================================"

# Test the deployment
echo "🧪 Testing deployment..."
sleep 30  # Wait for container to start

if curl -f "http://${FQDN}:8000/ping" > /dev/null 2>&1; then
    echo "✅ Container is responding!"
else
    echo "⚠️  Container might still be starting. Check logs with:"
    echo "   az container logs --resource-group ${RESOURCE_GROUP} --name ${CONTAINER_NAME}"
fi

echo ""
echo "📝 To view logs:"
echo "   az container logs --resource-group ${RESOURCE_GROUP} --name ${CONTAINER_NAME}"
echo ""
echo "🗑️  To delete the container:"
echo "   az container delete --resource-group ${RESOURCE_GROUP} --name ${CONTAINER_NAME}" 