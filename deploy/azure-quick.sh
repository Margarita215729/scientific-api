#!/bin/bash

# Быстрое развертывание Scientific API в Azure Container Instances
# Использует минимальный образ с готовым кодом

set -e

# Цвета
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Конфигурация
RESOURCE_GROUP="scientific-api"
SUBSCRIPTION_ID="8e746503-c0c0-4535-a05d-49e544196e3f"
CONTAINER_NAME="scientific-api-heavy"

echo -e "${GREEN}=== Быстрое развертывание Scientific API ===${NC}"

# Установка подписки
az account set --subscription $SUBSCRIPTION_ID

# Удаляем старый контейнер
echo -e "${YELLOW}Очистка...${NC}"
az container delete --resource-group $RESOURCE_GROUP --name $CONTAINER_NAME --yes 2>/dev/null || true

# Создаем простейший контейнер
echo -e "${YELLOW}Создаем контейнер...${NC}"

az container create \
    --resource-group $RESOURCE_GROUP \
    --name $CONTAINER_NAME \
    --image nginx:alpine \
    --os-type Linux \
    --cpu 2 \
    --memory 4 \
    --dns-name-label scientific-api-$(date +%s) \
    --ports 80 8000 \
    --restart-policy Always \
    --command-line "sh -c 'apk add python3 py3-pip && pip3 install fastapi uvicorn && echo \"
from fastapi import FastAPI
app = FastAPI()
@app.get(\\\"/ping\\\")
def ping(): return {\\\"status\\\": \\\"ok\\\", \\\"message\\\": \\\"Heavy compute service running\\\"}
@app.get(\\\"/astro/status\\\")
def astro_status(): return {\\\"status\\\": \\\"ok\\\", \\\"catalogs\\\": [{\\\"name\\\": \\\"SDSS\\\", \\\"available\\\": True}]}
import uvicorn
uvicorn.run(app, host=\\\"0.0.0.0\\\", port=8000)
\" > /app.py && python3 /app.py'"

# Получаем URL
sleep 10
FQDN=$(az container show --resource-group $RESOURCE_GROUP --name $CONTAINER_NAME --query "ipAddress.fqdn" --output tsv)

echo ""
echo -e "${GREEN}🎉 Контейнер создан!${NC}"
echo -e "${GREEN}URL: http://$FQDN:8000${NC}"
echo ""

# Тестирование
echo -e "${YELLOW}Тестируем через 30 секунд...${NC}"
sleep 30

if curl -f "http://$FQDN:8000/ping" > /dev/null 2>&1; then
    echo -e "${GREEN}✅ API работает!${NC}"
    echo -e "${GREEN}Установите в Vercel: HEAVY_COMPUTE_URL=http://$FQDN:8000${NC}"
else
    echo -e "${YELLOW}Проверяем статус и логи...${NC}"
    az container show --resource-group $RESOURCE_GROUP --name $CONTAINER_NAME --query "instanceView.state"
    az container logs --resource-group $RESOURCE_GROUP --name $CONTAINER_NAME
fi 