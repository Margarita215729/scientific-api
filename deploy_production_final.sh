#!/bin/bash

# Final Production Deployment Script for Scientific API
# This script performs a complete production deployment

set -e

echo "🚀 Scientific API - Final Production Deployment"
echo "=============================================="

# Set Azure CLI environment
export REQUESTS_CA_BUNDLE=/Users/Gret/scientific-api-1/venv/lib/python3.13/site-packages/certifi/cacert.pem

# Configuration
RESOURCE_GROUP="scientific-api"
APP_NAME="scientific-api"
LOCATION="canadacentral"
PLAN_NAME="ASP-scientificapi-8622"

echo "📋 Configuration:"
echo "  Resource Group: $RESOURCE_GROUP"
echo "  App Name: $APP_NAME"
echo "  Location: $LOCATION"
echo "  Plan Name: $PLAN_NAME"
echo ""

# Check Azure CLI
echo "🔍 Checking Azure CLI..."
if ! command -v az &> /dev/null; then
    echo "❌ Azure CLI not installed"
    exit 1
fi

# Check Azure login
echo "🔐 Checking Azure login..."
if ! az account show &> /dev/null; then
    echo "🔐 Please login to Azure..."
    az login
fi

echo "✅ Azure authentication successful"

# Check if app service plan exists
echo "🔍 Checking app service plan..."
if ! az appservice plan show --name "$PLAN_NAME" --resource-group $RESOURCE_GROUP &> /dev/null; then
    echo "📦 Creating app service plan..."
    az appservice plan create \
        --name "$PLAN_NAME" \
        --resource-group $RESOURCE_GROUP \
        --sku B1 \
        --is-linux \
        --location $LOCATION
    echo "✅ App service plan created"
else
    echo "✅ App service plan exists"
fi

# Check if web app exists
echo "🔍 Checking web app..."
if ! az webapp show --name $APP_NAME --resource-group $RESOURCE_GROUP &> /dev/null; then
    echo "📦 Creating web app..."
    az webapp create \
        --name $APP_NAME \
        --resource-group $RESOURCE_GROUP \
        --plan "$PLAN_NAME" \
        --deployment-container-image-name "gretk/scientific-api-app-image:scientific-api"
    echo "✅ Web app created"
else
    echo "✅ Web app exists"
fi

# Configure container settings
echo "🐳 Configuring container settings..."
az webapp config container set \
    --name $APP_NAME \
    --resource-group $RESOURCE_GROUP \
    --container-image-name "gretk/scientific-api-app-image:scientific-api"

echo "✅ Container settings configured"

# Configure all application settings
echo "⚙️  Configuring application settings..."

# Core settings
az webapp config appsettings set \
    --name $APP_NAME \
    --resource-group $RESOURCE_GROUP \
    --settings \
    ENVIRONMENT=production \
    PYTHONUNBUFFERED=1 \
    DEBUG=false \
    LOG_LEVEL=INFO \
    DISABLE_SSL_VERIFICATION=true \
    PYTHONHTTPSVERIFY=0

# API Keys (load from .env file)
echo "🔑 Loading API keys from .env file..."
if [ -f ".env" ]; then
    source .env
    az webapp config appsettings set \
        --name $APP_NAME \
        --resource-group $RESOURCE_GROUP \
        --settings \
        ADSABS_TOKEN="$ADSABS_TOKEN" \
        SERPAPI_KEY="$SERPAPI_KEY" \
        GOOGLE_CLIENT_ID="$GOOGLE_CLIENT_ID" \
        GOOGLE_CLIENT_SECRET="$GOOGLE_CLIENT_SECRET" \
        GOOGLE_REFRESH_TOKEN="$GOOGLE_REFRESH_TOKEN" \
        HUGGINGFACE_ACCESS_TOKEN="$HUGGINGFACE_ACCESS_TOKEN"
    echo "✅ API keys configured from .env file"
else
    echo "⚠️  .env file not found. Please configure API keys manually."
fi

# Database settings (load from .env file)
if [ -f ".env" ]; then
    az webapp config appsettings set \
        --name $APP_NAME \
        --resource-group $RESOURCE_GROUP \
        --settings \
        DB_TYPE="$DB_TYPE" \
        AZURE_COSMOS_CONNECTION_STRING="$AZURE_COSMOS_CONNECTION_STRING" \
        COSMOSDB_CONNECTION_STRING="$COSMOSDB_CONNECTION_STRING" \
        MONGODB_URI="$MONGODB_URI" \
        COSMOS_DB_ACCOUNT="$COSMOS_DB_ACCOUNT" \
        COSMOS_DB_DATABASE="$COSMOS_DB_DATABASE" \
        COSMOS_DB_CONTAINER="$COSMOS_DB_CONTAINER" \
        MONGODB_DATABASE_NAME="$MONGODB_DATABASE_NAME" \
        COSMOS_DATABASE_NAME="$COSMOS_DATABASE_NAME" \
        COSMOS_DB_ENDPOINT="$COSMOS_DB_ENDPOINT" \
        COSMOS_DB_KEY="$COSMOS_DB_KEY" \
        CACHE_TTL_HOURS="$CACHE_TTL_HOURS" \
        DATABASE_URL="$DATABASE_URL"
    echo "✅ Database settings configured from .env file"
fi

# Security settings (load from .env file)
if [ -f ".env" ]; then
    az webapp config appsettings set \
        --name $APP_NAME \
        --resource-group $RESOURCE_GROUP \
        --settings \
        ADMIN_API_KEY="$ADMIN_API_KEY" \
        USER_API_KEYS="$USER_API_KEYS" \
        RATE_LIMIT_REQUESTS="$RATE_LIMIT_REQUESTS" \
        RATE_LIMIT_WINDOW="$RATE_LIMIT_WINDOW"
    echo "✅ Security settings configured from .env file"
fi

# App settings
az webapp config appsettings set \
    --name $APP_NAME \
    --resource-group $RESOURCE_GROUP \
    --settings \
    WEBSITES_ENABLE_APP_SERVICE_STORAGE=false \
    WEBSITES_PORT=8000 \
    PORT=8000 \
    HEAVY_PIPELINE_ON_START=false \
    DEBUG_RELOAD=false \
    WEB_CONCURRENCY=1

echo "✅ All application settings configured"

# Restart the app
echo "🔄 Restarting web app..."
az webapp restart --name $APP_NAME --resource-group $RESOURCE_GROUP

# Get the app URL
echo "📋 Getting app information..."
APP_URL=$(az webapp show --name $APP_NAME --resource-group $RESOURCE_GROUP --query defaultHostName --output tsv)

echo ""
echo "✅ Deployment completed successfully!"
echo "===================================="
echo "🌐 App URL: https://$APP_URL"
echo "📚 API Documentation: https://$APP_URL/docs"
echo "🏥 Health Check: https://$APP_URL/ping"
echo "🔬 Research API: https://$APP_URL/api/research/status"
echo "🤖 ML Models: https://$APP_URL/api/ml/models"
echo "📊 Data Management: https://$APP_URL/api/data/status"
echo "===================================="

# Wait for app to start
echo "⏳ Waiting for application to start..."
sleep 60

# Test the deployment
echo "🧪 Testing deployment..."

# Test health check
if curl -f "https://$APP_URL/ping" > /dev/null 2>&1; then
    echo "✅ Health check passed!"
else
    echo "⚠️  Health check failed, but app might still be starting"
fi

# Test API documentation
if curl -f "https://$APP_URL/docs" > /dev/null 2>&1; then
    echo "✅ API documentation accessible!"
else
    echo "⚠️  API documentation not accessible yet"
fi

echo ""
echo "📝 Useful commands:"
echo "   View logs: az webapp log tail --name $APP_NAME --resource-group $RESOURCE_GROUP"
echo "   Stop app: az webapp stop --name $APP_NAME --resource-group $RESOURCE_GROUP"
echo "   Start app: az webapp start --name $APP_NAME --resource-group $RESOURCE_GROUP"
echo "   Restart app: az webapp restart --name $APP_NAME --resource-group $RESOURCE_GROUP"
echo ""
echo "🎉 Production deployment completed!"
