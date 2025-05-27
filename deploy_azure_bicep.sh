#!/bin/bash

# Azure Web App Deployment Script using Bicep
# Scientific API - Updated configuration with Bicep template

set -e

echo "🚀 Deploying Scientific API to Azure Web App using Bicep..."

# Configuration - Ensure these are correct or sourced from a secure location/CI variables
SUBSCRIPTION_ID="${AZURE_SUBSCRIPTION_ID:-8e746503-c0c0-4535-a05d-49e544196e3f}"
RESOURCE_GROUP="${AZURE_RESOURCE_GROUP:-scientific-api}"
APP_NAME="${AZURE_APP_NAME:-scientific-api}"
LOCATION="${AZURE_LOCATION:-canadacentral}"
HOSTING_PLAN="${AZURE_HOSTING_PLAN:-ASP-scientificapi-aef1}"
# Docker image should be pulled from your registry where it's pushed after build
# Example: DOCKER_IMAGE="youracr.azurecr.io/scientific-api-app-image:latest"
# Using the one from azure.env for now, assuming it's publicly available or registry is configured for pull
DOCKER_IMAGE="${DOCKER_IMAGE_FULL_PATH:-index.docker.io/gretk/scientific-api-app-image:scientific-api}"

# Cosmos DB Configuration for MongoDB API
AZURE_COSMOS_CONNECTION_STRING="${AZURE_COSMOS_CONNECTION_STRING:?Please set AZURE_COSMOS_CONNECTION_STRING environment variable}"
COSMOS_DATABASE_NAME="${COSMOS_DATABASE_NAME:-scientific-data}" # Убедитесь, что это имя совпадает с тем, что в Connection String или желаемым
DB_TYPE="${DB_TYPE:-cosmosdb_mongo}" 

ADSABS_TOKEN="${ADSABS_TOKEN:?Please set ADSABS_TOKEN environment variable}"
SERPAPI_KEY="${SERPAPI_KEY:?Please set SERPAPI_KEY environment variable}"


VNET_NAME="${VNET_NAME:-vnet-euoxdfir}"
SUBNET_NAME="${SUBNET_NAME:-subnet-nwivqmzl}"

# Flag to control heavy data pipeline on startup
HEAVY_PIPELINE_ON_START="${HEAVY_PIPELINE_ON_START:-true}" # Set to true to run preprocessor on start

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
echo "  Cosmos DB Connection String: YES (sensitive value - not shown)"
echo "  Cosmos DB Database Name: $COSMOS_DATABASE_NAME"
echo "  DB Type: $DB_TYPE"
echo "  VNet: $VNET_NAME"
echo "  Subnet: $SUBNET_NAME"
echo "  Run Heavy Pipeline on Start: $HEAVY_PIPELINE_ON_START"
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
az account set --subscription "$SUBSCRIPTION_ID"

# Check if resource group exists, create if not
echo -e "${YELLOW}📦 Checking resource group '$RESOURCE_GROUP'...${NC}"
if ! az group show --name "$RESOURCE_GROUP" &> /dev/null; then
    echo -e "${YELLOW}📦 Resource group '$RESOURCE_GROUP' not found. Creating...${NC}"
    az group create --name "$RESOURCE_GROUP" --location "$LOCATION"
    echo -e "${GREEN}✅ Resource group '$RESOURCE_GROUP' created.${NC}"
else
    echo -e "${GREEN}✅ Resource group '$RESOURCE_GROUP' already exists.${NC}"
fi

# Deploy using Bicep template
echo -e "${YELLOW}🚀 Deploying Web App using Bicep template 'azure-webapp-bicep.bicep'...${NC}"
az deployment group create \
    --resource-group "$RESOURCE_GROUP" \
    --template-file azure-webapp-bicep.bicep \
    --parameters \
        sites_scientific_api_name="$APP_NAME" \
        serverfarms_ASP_scientificapi_aef1_externalid="/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.Web/serverfarms/$HOSTING_PLAN" \
        virtualNetworks_vnet_euoxdfir_externalid="/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.Network/virtualNetworks/$VNET_NAME"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Bicep deployment completed successfully!${NC}"
else
    echo -e "${RED}❌ Bicep deployment failed.${NC}"
    exit 1
fi

# Configure app settings
# Ensure all necessary environment variables for the application are set here.
# Especially those for database connection and API keys.
echo -e "${YELLOW}⚙️  Configuring app settings for Web App '$APP_NAME'...${NC}"

# Создаем JSON массив для App Settings
APP_SETTINGS_JSON="["
APP_SETTINGS_JSON+="{\"name\": \"WEBSITES_ENABLE_APP_SERVICE_STORAGE\", \"value\": \"false\", \"slotSetting\": false},"
APP_SETTINGS_JSON+="{\"name\": \"DOCKER_REGISTRY_SERVER_URL\", \"value\": \"${DOCKER_IMAGE%%/*}\", \"slotSetting\": false},"
APP_SETTINGS_JSON+="{\"name\": \"DB_TYPE\", \"value\": \"${DB_TYPE}\", \"slotSetting\": false},"
APP_SETTINGS_JSON+="{\"name\": \"AZURE_COSMOS_CONNECTION_STRING\", \"value\": \"${AZURE_COSMOS_CONNECTION_STRING}\", \"slotSetting\": false},"
APP_SETTINGS_JSON+="{\"name\": \"COSMOS_DATABASE_NAME\", \"value\": \"${COSMOS_DATABASE_NAME}\", \"slotSetting\": false},"
APP_SETTINGS_JSON+="{\"name\": \"ADSABS_TOKEN\", \"value\": \"${ADSABS_TOKEN}\", \"slotSetting\": false},"
APP_SETTINGS_JSON+="{\"name\": \"SERPAPI_KEY\", \"value\": \"${SERPAPI_KEY}\", \"slotSetting\": false},"
APP_SETTINGS_JSON+="{\"name\": \"PYTHONUNBUFFERED\", \"value\": \"1\", \"slotSetting\": false},"
APP_SETTINGS_JSON+="{\"name\": \"ENVIRONMENT\", \"value\": \"production\", \"slotSetting\": false},"
APP_SETTINGS_JSON+="{\"name\": \"HEAVY_PIPELINE_ON_START\", \"value\": \"${HEAVY_PIPELINE_ON_START}\", \"slotSetting\": false}"
APP_SETTINGS_JSON+="]"

# Удаляем временный файл, если он существует
rm -f appsettings.json
# Записываем JSON в файл
echo "$APP_SETTINGS_JSON" > appsettings.json

# Устанавливаем App Settings из файла
az webapp config appsettings set \
    --resource-group "$RESOURCE_GROUP" \
    --name "$APP_NAME" \
    --settings "@appsettings.json" # Используем @ для указания файла

# Удаляем временный файл
rm -f appsettings.json

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ App settings configured successfully!${NC}"
else
    echo -e "${RED}❌ Failed to configure app settings.${NC}"
    # exit 1 # Decide if this is a fatal error
fi


# Get deployment information
echo -e "${YELLOW}📊 Getting deployment information...${NC}"
APP_URL=$(az webapp show --resource-group "$RESOURCE_GROUP" --name "$APP_NAME" --query "defaultHostName" --output tsv)
APP_STATUS=$(az webapp show --resource-group "$RESOURCE_GROUP" --name "$APP_NAME" --query "state" --output tsv)

if [ -z "$APP_URL" ]; then 
    echo -e "${RED}❌ Could not retrieve App URL. Deployment might have issues.${NC}"
else
    APP_URL="https://$APP_URL" # Prepend https
    echo ""
    echo -e "${GREEN}✅ Azure Web App deployment appears successful!${NC}"
    echo "==================================================" 
    echo -e "${GREEN}🌐 App URL: $APP_URL${NC}"
    echo -e "${GREEN}📊 Status: $APP_STATUS${NC}"
    echo -e "${GREEN}🏥 Health Check: $APP_URL/ping или $APP_URL/api/health ${NC}"
    echo -e "${GREEN}📚 API Docs: $APP_URL/docs${NC}"
    echo "=================================================="
    echo ""
    echo -e "${BLUE}🔒 Security Settings Reminder:${NC}"
    echo "  FTP Access: Disabled (as per Bicep)"
    echo "  SCM Access: Disabled (as per Bicep)"
    echo "  HTTPS Only: Enabled (as per Bicep)"
    echo "  TLS Version: 1.2+ (as per Bicep)"
    echo "  VNet Integration: Enabled for subnet '$SUBNET_NAME' (as per Bicep)"
    echo -e "  App Settings for secrets (AZURE_COSMOS_CONNECTION_STRING, ADSABS_TOKEN, etc.) should be reviewed in Azure Portal for security."
    echo ""

    # Test the deployment
    echo -e "${YELLOW}🧪 Testing deployment (waiting 60s for app to start...)${NC}"
    sleep 60

    HEALTH_ENDPOINT_TO_TEST="$APP_URL/api/health"
    echo "Attempting to curl $HEALTH_ENDPOINT_TO_TEST ..."
    if curl -f -L --connect-timeout 10 --max-time 20 "$HEALTH_ENDPOINT_TO_TEST" > /dev/null 2>&1; then
        echo -e "${GREEN}✅ App is responding! Output of health check: ${NC}"
        curl -L "$HEALTH_ENDPOINT_TO_TEST" # Print the actual health check output
        echo ""
    else
        echo -e "${RED}⚠️  App health check failed or timed out. Check logs in Azure Portal:${NC}"
        echo "   az webapp log tail --resource-group \"$RESOURCE_GROUP\" --name \"$APP_NAME\""
    fi
fi

echo ""
echo -e "${BLUE}📝 Management Commands:${NC}"
echo "  View logs: az webapp log tail --resource-group \"$RESOURCE_GROUP\" --name \"$APP_NAME\""
echo "  Restart app: az webapp restart --resource-group \"$RESOURCE_GROUP\" --name \"$APP_NAME\""
echo "  View config: az webapp config appsettings list --resource-group \"$RESOURCE_GROUP\" --name \"$APP_NAME\""
echo ""
echo -e "${GREEN}🎉 Deployment script finished! Your Scientific API should be running on Azure Web App.${NC}" 