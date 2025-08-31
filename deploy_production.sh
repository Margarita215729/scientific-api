#!/bin/bash

# Production deployment script for Scientific API
# This script builds and deploys the application to Azure

set -e

echo "🚀 Scientific API Production Deployment"
echo "======================================="

# Configuration
RESOURCE_GROUP="scientific-api"
APP_NAME="scientific-api"
LOCATION="canadacentral"
IMAGE_NAME="scientific-api-prod"
REGISTRY_NAME="scientificapiregistry"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

success() {
    echo -e "${GREEN}✅ $1${NC}"
}

warning() {
    echo -e "${YELLOW}⚠️ $1${NC}"
}

error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    if ! command -v az &> /dev/null; then
        error "Azure CLI is not installed. Please install it first."
        exit 1
    fi
    
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed. Please install it first."
        exit 1
    fi
    
    # Check if logged in to Azure
    if ! az account show &> /dev/null; then
        error "Not logged in to Azure. Please run 'az login' first."
        exit 1
    fi
    
    success "Prerequisites check passed"
}

# Build Docker image
build_image() {
    log "Building Docker image..."
    
    if docker build -t $IMAGE_NAME:latest .; then
        success "Docker image built successfully"
    else
        error "Failed to build Docker image"
        exit 1
    fi
}

# Create Azure Container Registry if it doesn't exist
create_registry() {
    log "Setting up Azure Container Registry..."
    
    # Check if registry exists
    if az acr show --name $REGISTRY_NAME --resource-group $RESOURCE_GROUP &> /dev/null; then
        success "Container registry already exists"
    else
        log "Creating container registry..."
        if az acr create --name $REGISTRY_NAME --resource-group $RESOURCE_GROUP --sku Basic --location $LOCATION; then
            success "Container registry created"
        else
            error "Failed to create container registry"
            exit 1
        fi
    fi
    
    # Enable admin user
    az acr update --name $REGISTRY_NAME --admin-enabled true
}

# Push image to Azure Container Registry
push_image() {
    log "Pushing image to Azure Container Registry..."
    
    # Login to registry
    az acr login --name $REGISTRY_NAME
    
    # Tag image
    REGISTRY_URL="${REGISTRY_NAME}.azurecr.io"
    docker tag $IMAGE_NAME:latest $REGISTRY_URL/$IMAGE_NAME:latest
    
    # Push image
    if docker push $REGISTRY_URL/$IMAGE_NAME:latest; then
        success "Image pushed successfully"
    else
        error "Failed to push image"
        exit 1
    fi
}

# Deploy to Azure Web App
deploy_webapp() {
    log "Deploying to Azure Web App..."
    
    # Get registry credentials
    REGISTRY_PASSWORD=$(az acr credential show --name $REGISTRY_NAME --query "passwords[0].value" --output tsv)
    REGISTRY_URL="${REGISTRY_NAME}.azurecr.io"
    
    # Check if app service plan exists
    if ! az appservice plan show --name "$APP_NAME-plan" --resource-group $RESOURCE_GROUP &> /dev/null; then
        log "Creating App Service Plan..."
        az appservice plan create --name "$APP_NAME-plan" --resource-group $RESOURCE_GROUP --sku B1 --is-linux
    fi
    
    # Check if web app exists
    if az webapp show --name $APP_NAME --resource-group $RESOURCE_GROUP &> /dev/null; then
        log "Updating existing web app..."
        az webapp config container set --name $APP_NAME --resource-group $RESOURCE_GROUP \
            --docker-custom-image-name "$REGISTRY_URL/$IMAGE_NAME:latest" \
            --docker-registry-server-url "https://$REGISTRY_URL" \
            --docker-registry-server-user $REGISTRY_NAME \
            --docker-registry-server-password "$REGISTRY_PASSWORD"
    else
        log "Creating new web app..."
        az webapp create --name $APP_NAME --resource-group $RESOURCE_GROUP \
            --plan "$APP_NAME-plan" \
            --deployment-container-image-name "$REGISTRY_URL/$IMAGE_NAME:latest" \
            --docker-registry-server-url "https://$REGISTRY_URL" \
            --docker-registry-server-user $REGISTRY_NAME \
            --docker-registry-server-password "$REGISTRY_PASSWORD"
    fi
    
    # Configure app settings
    log "Configuring app settings..."
    az webapp config appsettings set --name $APP_NAME --resource-group $RESOURCE_GROUP \
        --settings \
        ENVIRONMENT=production \
        PYTHONUNBUFFERED=1 \
        PORT=8000 \
        WEBSITES_ENABLE_APP_SERVICE_STORAGE=false \
        WEBSITES_PORT=8000
    
    success "Web app deployed successfully"
}

# Configure custom domain and SSL (optional)
configure_domain() {
    if [ -n "$CUSTOM_DOMAIN" ]; then
        log "Configuring custom domain: $CUSTOM_DOMAIN"
        # Add custom domain configuration here if needed
        warning "Custom domain configuration not implemented in this script"
    fi
}

# Run health check
health_check() {
    log "Running health check..."
    
    APP_URL="https://$APP_NAME.azurewebsites.net"
    
    # Wait for deployment to complete
    sleep 30
    
    # Check if app is responding
    if curl -f "$APP_URL/ping" &> /dev/null; then
        success "Health check passed - application is running"
        echo "🌐 Application URL: $APP_URL"
        echo "📚 API Documentation: $APP_URL/docs"
    else
        warning "Health check failed - application may still be starting up"
        echo "Please check the application logs in Azure portal"
    fi
}

# Main deployment process
main() {
    echo "Starting deployment process..."
    
    check_prerequisites
    build_image
    create_registry
    push_image
    deploy_webapp
    configure_domain
    health_check
    
    echo ""
    success "🎉 Deployment completed successfully!"
    echo "Application URL: https://$APP_NAME.azurewebsites.net"
    echo "API Documentation: https://$APP_NAME.azurewebsites.net/docs"
    echo ""
    echo "To monitor the application:"
    echo "- Azure Portal: https://portal.azure.com"
    echo "- Resource Group: $RESOURCE_GROUP"
    echo "- App Service: $APP_NAME"
}

# Handle script arguments
case "${1:-deploy}" in
    "build")
        check_prerequisites
        build_image
        ;;
    "push")
        check_prerequisites
        create_registry
        push_image
        ;;
    "deploy")
        main
        ;;
    "health")
        health_check
        ;;
    *)
        echo "Usage: $0 [build|push|deploy|health]"
        echo "  build  - Build Docker image only"
        echo "  push   - Build and push to registry"
        echo "  deploy - Full deployment (default)"
        echo "  health - Run health check only"
        exit 1
        ;;
esac
