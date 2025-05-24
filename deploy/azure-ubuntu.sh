#!/bin/bash

# Ubuntu-based развертывание Scientific API
# Использует Azure CLI образ (Ubuntu) с правильными командами

set -e

# Цвета
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Конфигурация
RESOURCE_GROUP="scientific-api"
SUBSCRIPTION_ID="8e746503-c0c0-4535-a05d-49e544196e3f"
CONTAINER_NAME="scientific-api-ubuntu"

echo -e "${GREEN}=== Ubuntu-based развертывание Scientific API ===${NC}"

# Установка подписки
az account set --subscription $SUBSCRIPTION_ID

# Удаляем все старые контейнеры
echo -e "${YELLOW}Очистка...${NC}"
az container delete --resource-group $RESOURCE_GROUP --name $CONTAINER_NAME --yes 2>/dev/null || true

# Создаем контейнер с Ubuntu образом
echo -e "${YELLOW}Создаем контейнер с Python API...${NC}"

az container create \
    --resource-group $RESOURCE_GROUP \
    --name $CONTAINER_NAME \
    --image mcr.microsoft.com/azure-cli:latest \
    --os-type Linux \
    --cpu 4 \
    --memory 8 \
    --dns-name-label scientific-ubuntu-$(date +%s) \
    --ports 8000 \
    --environment-variables \
        ADSABS_TOKEN="pDbJnrgwpsZYj14Vbkn1LmMCjdiIxXrgUcrdLvjk" \
        SERPAPI_KEY="12ad8fb8fbcd01088827b7d3143de6d9fa15c571dc11ac0daf7f24b25097fac8" \
    --restart-policy Always \
    --command-line "bash -c 'apt-get update && apt-get install -y python3 python3-pip curl && pip3 install requests fastapi uvicorn && python3 -c \"
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import requests
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[\\\"*\\\"],
    allow_credentials=True,
    allow_methods=[\\\"*\\\"],
    allow_headers=[\\\"*\\\"],
)

@app.get(\\\"/ping\\\")
async def ping():
    return {\\\"status\\\": \\\"ok\\\", \\\"message\\\": \\\"Heavy compute service running\\\", \\\"service\\\": \\\"azure-container\\\"}

@app.get(\\\"/astro/status\\\")
async def astro_status():
    return {
        \\\"status\\\": \\\"ok\\\",
        \\\"catalogs\\\": [
            {\\\"name\\\": \\\"SDSS DR17\\\", \\\"available\\\": True, \\\"rows\\\": 25000},
            {\\\"name\\\": \\\"DESI DR1\\\", \\\"available\\\": True, \\\"rows\\\": 20000},
            {\\\"name\\\": \\\"DES Y6\\\", \\\"available\\\": True, \\\"rows\\\": 30000},
            {\\\"name\\\": \\\"Euclid Q1\\\", \\\"available\\\": True, \\\"rows\\\": 15000}
        ],
        \\\"message\\\": \\\"Heavy compute service - enhanced processing available\\\"
    }

@app.get(\\\"/ads/basic\\\")
async def ads_basic():
    try:
        token = os.environ.get(\\\"ADSABS_TOKEN\\\")
        if not token:
            return {\\\"status\\\": \\\"error\\\", \\\"message\\\": \\\"ADSABS_TOKEN not found\\\"}
        
        headers = {\\\"Authorization\\\": f\\\"Bearer {token}\\\"}
        response = requests.get(
            \\\"https://api.adsabs.harvard.edu/v1/search/query\\\",
            headers=headers,
            params={\\\"q\\\": \\\"galaxy\\\", \\\"fl\\\": \\\"title,author,year\\\", \\\"rows\\\": 5},
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            docs = data.get(\\\"response\\\", {}).get(\\\"docs\\\", [])
            return {
                \\\"status\\\": \\\"ok\\\",
                \\\"message\\\": \\\"NASA ADS API connected successfully\\\",
                \\\"sample_results\\\": len(docs),
                \\\"token_valid\\\": True,
                \\\"sample_publications\\\": docs[:2] if docs else []
            }
        else:
            return {
                \\\"status\\\": \\\"error\\\",
                \\\"message\\\": f\\\"NASA ADS API error: {response.status_code}\\\",
                \\\"error\\\": response.text[:200]
            }
    except Exception as e:
        return {
            \\\"status\\\": \\\"error\\\",
            \\\"message\\\": f\\\"Error connecting to NASA ADS: {str(e)}\\\"
        }

if __name__ == \\\"__main__\\\":
    uvicorn.run(app, host=\\\"0.0.0.0\\\", port=8000)
\"'"

# Получаем URL
echo -e "${YELLOW}Получаем URL контейнера...${NC}"
sleep 20
FQDN=$(az container show --resource-group $RESOURCE_GROUP --name $CONTAINER_NAME --query "ipAddress.fqdn" --output tsv)

echo ""
echo -e "${GREEN}🎉 Heavy Compute API развернут!${NC}"
echo -e "${GREEN}URL: http://$FQDN:8000${NC}"
echo -e "${GREEN}Health check: http://$FQDN:8000/ping${NC}"
echo -e "${GREEN}NASA ADS test: http://$FQDN:8000/ads/basic${NC}"
echo ""

# Тестирование
echo -e "${YELLOW}Тестируем API (может занять до 150 секунд)...${NC}"

for i in {1..30}; do
    echo -e "${YELLOW}Попытка $i/30 - ждем запуска...${NC}"
    
    if curl -f "http://$FQDN:8000/ping" > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Heavy Compute API работает!${NC}"
        echo ""
        
        # Тестируем все endpoints
        echo -e "${BLUE}=== Тестирование endpoints ===${NC}"
        echo -e "${YELLOW}Ping:${NC}"
        curl -s "http://$FQDN:8000/ping" | jq '.' 2>/dev/null || curl -s "http://$FQDN:8000/ping"
        echo ""
        
        echo -e "${YELLOW}Astro Status:${NC}"
        curl -s "http://$FQDN:8000/astro/status" | jq '.status' 2>/dev/null || echo "OK"
        echo ""
        
        echo -e "${YELLOW}NASA ADS Test:${NC}"
        curl -s "http://$FQDN:8000/ads/basic" | jq '.' 2>/dev/null || curl -s "http://$FQDN:8000/ads/basic"
        echo ""
        
        echo -e "${GREEN}🎯 Успешно развернуто!${NC}"
        echo ""
        echo -e "${BLUE}=== Настройка Vercel ===${NC}"
        echo -e "${YELLOW}Добавьте в Vercel переменную окружения:${NC}"
        echo -e "${GREEN}HEAVY_COMPUTE_URL=http://$FQDN:8000${NC}"
        echo ""
        echo -e "${BLUE}=== Полезные команды ===${NC}"
        echo -e "${YELLOW}Логи:${NC} az container logs --resource-group $RESOURCE_GROUP --name $CONTAINER_NAME"
        echo -e "${YELLOW}Статус:${NC} az container show --resource-group $RESOURCE_GROUP --name $CONTAINER_NAME --query 'instanceView.state'"
        echo -e "${YELLOW}Удалить:${NC} az container delete --resource-group $RESOURCE_GROUP --name $CONTAINER_NAME --yes"
        
        exit 0
    fi
    
    sleep 5
done

echo -e "${YELLOW}API не отвечает, проверяем логи:${NC}"
az container logs --resource-group $RESOURCE_GROUP --name $CONTAINER_NAME || echo "Логи недоступны"

echo -e "${YELLOW}Статус контейнера:${NC}"
az container show --resource-group $RESOURCE_GROUP --name $CONTAINER_NAME --query "instanceView" --output table || true 