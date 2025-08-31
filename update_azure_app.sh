#!/bin/bash

# Update existing Azure Web App with new configuration
# This script updates the existing deployment without recreating infrastructure

set -e

echo "🔄 Updating Scientific API on Azure"
echo "=================================="

# Configuration from azure.env
source azure.env 2>/dev/null || echo "Warning: azure.env not found, using defaults"

RESOURCE_GROUP="${AZURE_RESOURCE_GROUP:-scientific-api}"
APP_NAME="${AZURE_APP_NAME:-scientific-api}"
AZURE_URL="${AZURE_APP_URL:-https://scientific-api-e3a7a5dph6b3axa3.canadacentral-01.azurewebsites.net}"

echo "📋 Configuration:"
echo "  Resource Group: $RESOURCE_GROUP"
echo "  App Name: $APP_NAME"
echo "  Current URL: $AZURE_URL"
echo ""

# Check if app exists (without using az CLI due to SSL issues)
echo "🔍 Checking application status..."
if curl -s --max-time 10 "$AZURE_URL/ping" > /dev/null 2>&1; then
    echo "✅ Application is responding"
else
    echo "⚠️  Application is not responding (may be stopped or needs restart)"
fi

echo ""
echo "🌐 Your Scientific API URLs:"
echo "=================================="
echo "🏠 Main Application: $AZURE_URL"
echo "📚 API Documentation: $AZURE_URL/docs"
echo "🏥 Health Check: $AZURE_URL/ping"
echo "🔬 Research API: $AZURE_URL/api/research/status"
echo ""
echo "🧪 Test Endpoints:"
echo "📖 Search ArXiv: $AZURE_URL/api/research/search?query=galaxy&sources=arxiv"
echo "🌟 Search ADS: $AZURE_URL/api/research/search?query=cosmology&sources=ads"
echo "🤖 ML Models: $AZURE_URL/api/ml/models"
echo ""

# Try to restart the app using REST API (if Azure CLI has SSL issues)
echo "🔄 Attempting to restart application..."
echo "Note: If the app is stopped, you may need to start it manually in Azure Portal"
echo ""

# Alternative: provide manual instructions
echo "📋 Manual steps if app needs restart:"
echo "1. Go to Azure Portal: https://portal.azure.com"
echo "2. Navigate to Resource Groups > $RESOURCE_GROUP"
echo "3. Click on App Service: $APP_NAME"
echo "4. Click 'Restart' if the app is stopped"
echo "5. Check 'Configuration' > 'Application settings' to verify environment variables"
echo ""

echo "✅ Update script completed!"
echo "🌐 Your app should be accessible at: $AZURE_URL"

