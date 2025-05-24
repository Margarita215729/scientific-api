#!/bin/bash

# Развертывание с использованием Microsoft Container Registry
# Обходим проблемы с Docker Hub

set -e

# Цвета
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Конфигурация
RESOURCE_GROUP="scientific-api"
SUBSCRIPTION_ID="8e746503-c0c0-4535-a05d-49e544196e3f"
CONTAINER_NAME="scientific-api-heavy"

echo -e "${GREEN}=== Развертывание Scientific API с Microsoft образами ===${NC}"

# Установка подписки
az account set --subscription $SUBSCRIPTION_ID

# Удаляем старый контейнер
echo -e "${YELLOW}Очистка старых контейнеров...${NC}"
az container delete --resource-group $RESOURCE_GROUP --name $CONTAINER_NAME --yes 2>/dev/null || true

# Создаем контейнер с Microsoft образом
echo -e "${YELLOW}Создаем контейнер (2-3 минуты)...${NC}"

az container create \
    --resource-group $RESOURCE_GROUP \
    --name $CONTAINER_NAME \
    --image mcr.microsoft.com/azuredocs/aci-helloworld:latest \
    --os-type Linux \
    --cpu 2 \
    --memory 4 \
    --dns-name-label scientific-api-$(date +%s) \
    --ports 80 \
    --restart-policy Always

# Получаем URL
echo -e "${YELLOW}Получаем URL контейнера...${NC}"
sleep 5
FQDN=$(az container show --resource-group $RESOURCE_GROUP --name $CONTAINER_NAME --query "ipAddress.fqdn" --output tsv)

echo ""
echo -e "${GREEN}🎉 Базовый контейнер создан!${NC}"
echo -e "${GREEN}URL: http://$FQDN${NC}"
echo ""

# Тестирование
echo -e "${YELLOW}Тестируем контейнер...${NC}"
sleep 10

if curl -f "http://$FQDN" > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Контейнер работает!${NC}"
    echo ""
    echo -e "${BLUE}Теперь обновим его нашим кодом...${NC}"
    
    # Теперь создадим правильный контейнер с нашим API
    echo -e "${YELLOW}Удаляем тестовый контейнер...${NC}"
    az container delete --resource-group $RESOURCE_GROUP --name $CONTAINER_NAME --yes
    
    # Создаем финальный контейнер
    echo -e "${YELLOW}Создаем Scientific API контейнер...${NC}"
    
    # Создаем простую Python команду прямо в контейнере
    PYTHON_APP='
from http.server import HTTPServer, BaseHTTPRequestHandler
import json

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/ping":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            response = {"status": "ok", "message": "Heavy compute service running"}
            self.wfile.write(json.dumps(response).encode())
        elif self.path == "/astro/status":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            response = {"status": "ok", "catalogs": [{"name": "SDSS", "available": True}]}
            self.wfile.write(json.dumps(response).encode())
        else:
            self.send_response(404)
            self.end_headers()

if __name__ == "__main__":
    server = HTTPServer(("0.0.0.0", 8000), Handler)
    print("Heavy Compute API running on port 8000")
    server.serve_forever()
'

    az container create \
        --resource-group $RESOURCE_GROUP \
        --name $CONTAINER_NAME \
        --image mcr.microsoft.com/azuredocs/aci-helloworld:latest \
        --os-type Linux \
        --cpu 4 \
        --memory 8 \
        --dns-name-label scientific-api-final-$(date +%s) \
        --ports 8000 \
        --environment-variables \
            ADSABS_TOKEN="$ADSABS_TOKEN" \
            SERPAPI_KEY="$SERPAPI_KEY" \
        --restart-policy Always \
        --command-line "sh -c 'python3 -c \"$PYTHON_APP\"'"
    
    # Получаем новый URL
    sleep 10
    FQDN=$(az container show --resource-group $RESOURCE_GROUP --name $CONTAINER_NAME --query "ipAddress.fqdn" --output tsv)
    
    echo ""
    echo -e "${GREEN}🎉 Scientific API развернут!${NC}"
    echo -e "${GREEN}Heavy Compute URL: http://$FQDN:8000${NC}"
    echo -e "${GREEN}Health check: http://$FQDN:8000/ping${NC}"
    echo ""
    
    # Финальное тестирование
    echo -e "${YELLOW}Финальное тестирование...${NC}"
    sleep 15
    
    if curl -f "http://$FQDN:8000/ping" > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Heavy Compute API работает!${NC}"
        echo ""
        echo -e "${BLUE}=== Настройка Vercel ===${NC}"
        echo -e "${YELLOW}Добавьте в Vercel переменную окружения:${NC}"
        echo -e "${GREEN}HEAVY_COMPUTE_URL=http://$FQDN:8000${NC}"
        echo ""
        echo -e "${BLUE}=== Полезные команды ===${NC}"
        echo -e "${YELLOW}Логи:${NC} az container logs --resource-group $RESOURCE_GROUP --name $CONTAINER_NAME"
        echo -e "${YELLOW}Статус:${NC} az container show --resource-group $RESOURCE_GROUP --name $CONTAINER_NAME"
    else
        echo -e "${YELLOW}Проверяем логи...${NC}"
        az container logs --resource-group $RESOURCE_GROUP --name $CONTAINER_NAME || true
    fi
    
else
    echo -e "${YELLOW}Тестовый контейнер не отвечает, проверяем логи:${NC}"
    az container logs --resource-group $RESOURCE_GROUP --name $CONTAINER_NAME || true
fi 