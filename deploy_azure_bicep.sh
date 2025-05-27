#!/bin/bash

# Azure Web App Deployment Script using Bicep
# Scientific API - Updated configuration with Bicep template

set -e

echo "🚀 Deploying Scientific API to Azure Web App using Bicep..."

# Configuration from Bicep template
SUBSCRIPTION_ID="8e746503-c0c0-4535-a05d-49e544196e3f"
RESOURCE_GROUP="scientific-api"
APP_NAME="scientific-api"
LOCATION="canadacentral"
HOSTING_PLAN="ASP-scientificapi-aef1"
DOCKER_IMAGE="index.docker.io/gretk/scientific-api-app-image:scientific-api"
COSMOS_DB_ACCOUNT="scientific-api-server"
COSMOS_DB_DATABASE="scientific-data"
SUBNET_NAME="subnet-nwivqmzl"
VNET_NAME="vnet-euoxdfir"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}📋 Configuration:${NC}"
echo "  Subscription: $SUBSCRIPTION_ID"
echo "  Resource Group: $RESOURCE_GROUP"
echo "  App Name: $APP_NAME"
echo "  Location: $LOCATION"
echo "  Docker Image: $DOCKER_IMAGE"
echo "  Cosmos DB Account: $COSMOS_DB_ACCOUNT"
echo "  Cosmos DB Database: $COSMOS_DB_DATABASE"
echo "  VNet: $VNET_NAME"
echo "  Subnet: $SUBNET_NAME"
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

# Set subscription
echo -e "${YELLOW}🔧 Setting subscription...${NC}"
az account set --subscription $SUBSCRIPTION_ID

# Check if resource group exists
echo -e "${YELLOW}📦 Checking resource group...${NC}"
if ! az group show --name $RESOURCE_GROUP &> /dev/null; then
    echo -e "${YELLOW}📦 Creating resource group...${NC}"
    az group create --name $RESOURCE_GROUP --location $LOCATION
fi

# Deploy using Bicep template
echo -e "${YELLOW}🚀 Deploying Web App using Bicep template...${NC}"
az deployment group create \
    --resource-group $RESOURCE_GROUP \
    --template-file azure-webapp-bicep.bicep \
    --parameters \
        sites_scientific_api_name=$APP_NAME \
        serverfarms_ASP_scientificapi_aef1_externalid="/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.Web/serverfarms/$HOSTING_PLAN" \
        virtualNetworks_vnet_euoxdfir_externalid="/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.Network/virtualNetworks/$VNET_NAME"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Deployment completed successfully!${NC}"
else
    echo -e "${RED}❌ Deployment failed${NC}"
    exit 1
fi

# Configure app settings
echo -e "${YELLOW}⚙️  Configuring app settings...${NC}"
az webapp config appsettings set \
    --resource-group $RESOURCE_GROUP \
    --name $APP_NAME \
    --settings \
        WEBSITES_ENABLE_APP_SERVICE_STORAGE=false \
        DOCKER_REGISTRY_SERVER_URL=https://index.docker.io \
        COSMOS_DB_DATABASE=$COSMOS_DB_DATABASE \
        COSMOS_DB_ACCOUNT=$COSMOS_DB_ACCOUNT \
        SUBNET_NAME=$SUBNET_NAME \
        VNET_NAME=$VNET_NAME \
        PYTHONUNBUFFERED=1 \
        ENVIRONMENT=production

# Get deployment information
echo -e "${YELLOW}📊 Getting deployment information...${NC}"
APP_URL="https://$APP_NAME-e3a7a5dph6b3axa3.canadacentral-01.azurewebsites.net"
APP_STATUS=$(az webapp show --resource-group $RESOURCE_GROUP --name $APP_NAME --query "state" --output tsv)

echo ""
echo -e "${GREEN}✅ Azure Web App deployment completed!${NC}"
echo "============================================"
echo -e "${GREEN}🌐 App URL: $APP_URL${NC}"
echo -e "${GREEN}📊 Status: $APP_STATUS${NC}"
echo -e "${GREEN}🏥 Health Check: $APP_URL/ping${NC}"
echo -e "${GREEN}📚 API Docs: $APP_URL/docs${NC}"
echo -e "${GREEN}🔍 ADS Search: $APP_URL/ads${NC}"
echo "============================================"

# Security settings info
echo -e "${BLUE}🔒 Security Settings:${NC}"
echo "  FTP Access: Disabled"
echo "  SCM Access: Disabled"
echo "  HTTPS Only: Enabled"
echo "  TLS Version: 1.2+"
echo "  VNet Integration: Enabled ($SUBNET_NAME)"
echo ""

# Test the deployment
echo -e "${YELLOW}🧪 Testing deployment...${NC}"
echo "Waiting for app to start (this may take a few minutes)..."
sleep 120

if curl -f "$APP_URL/ping" > /dev/null 2>&1; then
    echo -e "${GREEN}✅ App is responding!${NC}"
else
    echo -e "${YELLOW}⚠️  App might still be starting. Check logs with:${NC}"
    echo "   az webapp log tail --resource-group $RESOURCE_GROUP --name $APP_NAME"
fi

echo ""
echo -e "${BLUE}📝 Management Commands:${NC}"
echo "  View logs: az webapp log tail --resource-group $RESOURCE_GROUP --name $APP_NAME"
echo "  Restart app: az webapp restart --resource-group $RESOURCE_GROUP --name $APP_NAME"
echo "  Stop app: az webapp stop --resource-group $RESOURCE_GROUP --name $APP_NAME"
echo "  Start app: az webapp start --resource-group $RESOURCE_GROUP --name $APP_NAME"
echo "  View config: az webapp config show --resource-group $RESOURCE_GROUP --name $APP_NAME"
echo ""
echo -e "${GREEN}🎉 Deployment complete! Your Scientific API is now running on Azure Web App with enhanced security.${NC}" 