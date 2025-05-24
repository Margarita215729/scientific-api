#!/bin/bash

# Оптимизированное развертывание для бесплатной подписки Azure
# Использует минимальные ресурсы: 1 CPU, 1.5GB RAM

set -e  # Выход при любой ошибке
set -x  # Включаем подробное логирование команд

# Цвета
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Конфигурация  
RESOURCE_GROUP="scientific-api"
SUBSCRIPTION_ID="8e746503-c0c0-4535-a05d-49e544196e3f" # Убедитесь, что это ваша актуальная подписка
CONTAINER_NAME="scientific-api-free"
LOCATION="eastus" # Укажите ваш регион, если отличается

# Образ Docker
# Используем Alpine для минимизации размера. 
# Если возникают ошибки RegistryErrorResponse, это может быть связано с лимитами Docker Hub.
# Рекомендуется использовать Azure Container Registry (ACR) для большей надежности.
# 1. Создайте ACR: az acr create --resource-group $RESOURCE_GROUP --name <yourACRname> --sku Basic --admin-enabled true
# 2. Войдите в ACR: az acr login --name <yourACRname>
# 3. Соберите и отправьте ваш образ в ACR (пример):
#    - docker build -t <yourACRname>.azurecr.io/$CONTAINER_NAME:latest .
#    - docker push <yourACRname>.azurecr.io/$CONTAINER_NAME:latest
# 4. Замените DOCKER_IMAGE на <yourACRname>.azurecr.io/$CONTAINER_NAME:latest ниже
#    и при необходимости передайте --registry-login-server <yourACRname>.azurecr.io --registry-username <yourACRname> --registry-password $(az acr credential show --name <yourACRname> --query passwords[0].value -o tsv)
DOCKER_IMAGE="python:3.11-alpine"

echo -e "${GREEN}=== Подробное развертывание для бесплатной подписки Azure ===${NC}"
echo -e "${YELLOW}Используем ресурсы: 1 CPU, 1.5GB RAM в регионе $LOCATION ${NC}"
echo -e "${YELLOW}Целевой образ Docker: $DOCKER_IMAGE ${NC}"

# Установка подписки
echo -e "${BLUE}Шаг 1: Установка активной подписки Azure...${NC}"
az account set --subscription $SUBSCRIPTION_ID
echo -e "${GREEN}Подписка установлена: $SUBSCRIPTION_ID${NC}"

# Проверка и создание группы ресурсов (если не существует)
echo -e "${BLUE}Шаг 2: Проверка/создание группы ресурсов $RESOURCE_GROUP в $LOCATION...${NC}"
if az group show --name $RESOURCE_GROUP &>/dev/null; then
    echo -e "${GREEN}Группа ресурсов $RESOURCE_GROUP уже существует.${NC}"
else
    echo -e "${YELLOW}Создание группы ресурсов $RESOURCE_GROUP в $LOCATION...${NC}"
    az group create --name $RESOURCE_GROUP --location $LOCATION
    echo -e "${GREEN}Группа ресурсов $RESOURCE_GROUP создана.${NC}"
fi

# Удаляем все старые контейнеры в resource group
echo -e "${BLUE}Шаг 3: Очистка всех старых контейнеров с именем $CONTAINER_NAME в группе $RESOURCE_GROUP...${NC}"
EXISTING_CONTAINER=$(az container show --resource-group $RESOURCE_GROUP --name $CONTAINER_NAME --query name -o tsv 2>/dev/null || true)
if [ ! -z "$EXISTING_CONTAINER" ]; then
    echo -e "${YELLOW}Удаляем существующий контейнер: $CONTAINER_NAME...${NC}"
    az container delete --resource-group $RESOURCE_GROUP --name $CONTAINER_NAME --yes
    echo -e "${GREEN}Контейнер $CONTAINER_NAME удален. Ожидание завершения (30 сек)...${NC}"
    sleep 30
else
    echo -e "${GREEN}Контейнер $CONTAINER_NAME не найден, очистка не требуется.${NC}"
fi

# Проверяем текущее использование квоты (информационно)
echo -e "${BLUE}Шаг 4: Проверка текущего использования контейнеров в группе $RESOURCE_GROUP (информационно)...${NC}"
az container list --resource-group $RESOURCE_GROUP --output table

# Создаем оптимизированный контейнер
echo -e "${BLUE}Шаг 5: Создание легковесного контейнера $CONTAINER_NAME...${NC}"
echo -e "${YELLOW}Это может занять несколько минут, особенно при первой загрузке образа $DOCKER_IMAGE.${NC}"

# Обратите внимание на --command-line: все кавычки и специальные символы должны быть правильно экранированы.
# Код Python передается как одна длинная строка.
COMMAND_LINE_SCRIPT='pip install --no-cache-dir fastapi uvicorn requests && python3 -c "
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import requests
import os
import json
from typing import Optional, List, Dict

app = FastAPI(title=\\\"Scientific API - Free Tier (Enhanced)\\\", version=\\\"1.1.0\\\")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[\"*\"] # Разрешаем все источники для простоты
    allow_credentials=True,
    allow_methods=[\"*\"],
    allow_headers=[\"*\"],
)

# Пример "реального" датасета (небольшой, для демонстрации)
# В реальном приложении это могло бы быть чтение из файла или небольшой БД
ASTRO_DATA = [
    {\"id\": \"SDSSJ000000.00+000000.0\", \"ra\": 0.0, \"dec\": 0.0, \"redshift\": 0.1, \"type\": \"GALAXY\", \"magnitude\": 18.5},
    {\"id\": \"SDSSJ010000.00+100000.0\", \"ra\": 15.0, \"dec\": 10.0, \"redshift\": 0.25, \"type\": \"GALAXY\", \"magnitude\": 19.2},
    {\"id\": \"SDSSJ020000.00-050000.0\", \"ra\": 30.0, \"dec\": -5.0, \"redshift\": 0.05, \"type\": \"STAR\", \"magnitude\": 15.0},
    {\"id\": \"SDSSJ030000.00+150000.0\", \"ra\": 45.0, \"dec\": 15.0, \"redshift\": 0.5, \"type\": \"QSO\", \"magnitude\": 20.0},
]

@app.get(\"/ping\")
async def ping():
    return {
        \"status\": \"ok\", 
        \"message\": \"Scientific API - Free Tier (Enhanced) is running!\",
        \"service_details\": \"Azure Container Instance on Free Tier configuration\",
        \"resources_config\": \"1 CPU, 1.5GB RAM\"
    }

@app.get(\"/astro/objects\")
async def get_astro_objects(
    object_type: Optional[str] = Query(None, description=\"Filter by object type (GALAXY, STAR, QSO)\"),
    min_redshift: Optional[float] = Query(None, description=\"Minimum redshift\"),
    limit: int = Query(10, ge=1, le=100, description=\"Number of results to return\")
) -> List[Dict]:
    \"\"\" Возвращает список астрономических объектов с фильтрацией. \"\"\"
    filtered_data = ASTRO_DATA
    if object_type:
        filtered_data = [obj for obj in filtered_data if obj[\"type\"] == object_type.upper()]
    if min_redshift is not None:
        filtered_data = [obj for obj in filtered_data if obj[\"redshift\"] >= min_redshift]
    
    return filtered_data[:limit]

@app.get(\"/astro/objects/{object_id}\")
async def get_astro_object_by_id(object_id: str) -> Dict:
    \"\"\" Возвращает астрономический объект по его ID. \"\"\"
    for obj in ASTRO_DATA:
        if obj[\"id\"] == object_id:
            return obj
    return {\"error\": \"Object not found\", \"object_id\": object_id}

# Эмуляция более сложной обработки данных
@app.post(\"/astro/analyze\")
async def analyze_data(data: List[Dict]) -> Dict:
    \"\"\" Пример эндпоинта для \'реальной\' обработки. 
        Здесь могла бы быть более сложная логика.
    \"\"\"
    if not data:
        return {\"error\": \"No data provided for analysis\"}
    
    num_objects = len(data)
    avg_magnitude = sum(d.get(\"magnitude\", 25.0) for d in data) / num_objects if num_objects > 0 else 0
    redshift_values = [d.get(\"redshift\") for d in data if d.get(\"redshift\") is not None]
    avg_redshift = sum(redshift_values) / len(redshift_values) if redshift_values else 0

    return {
        \"analysis_summary\": {
            \"received_objects\": num_objects,
            \"average_magnitude\": round(avg_magnitude, 2),
            \"average_redshift\": round(avg_redshift, 3),
            \"status\": \"Analysis complete (simulated)\"
        }
    }

@app.get(\"/ads/basic\")
async def ads_basic(
    query: str = Query(\"galaxy\", description=\"Search query for NASA ADS\"),
    rows: int = Query(3, ge=1, le=10, description=\"Number of results from ADS\")
):
    \"\"\" Запрос к NASA ADS API с ограниченными результатами для Free Tier. \"\"\"
    try:
        token = os.environ.get(\"ADSABS_TOKEN\")
        if not token:
            return {\"status\": \"error\", \"message\": \"ADSABS_TOKEN environment variable not found\"}
        
        headers = {\"Authorization\": f\"Bearer {token}\"}
        params = {\"q\": query, \"fl\": \"title,author,year,pub\", \"rows\": rows}
        
        ads_url = \"https://api.adsabs.harvard.edu/v1/search/query\"
        response = requests.get(ads_url, headers=headers, params=params, timeout=15)
        
        response.raise_for_status() # Вызовет исключение для HTTP ошибок 4xx/5xx
        
        data = response.json()
        docs = data.get(\"response\", {}).get(\"docs\", [])
        return {
            \"status\": \"ok\",
            \"message\": \"NASA ADS API connected successfully - Free tier results\",
            \"query_sent\": query,
            \"results_count\": len(docs),
            \"results\": docs,
            \"note\": f\"Limited to {rows} results in free tier.\"
        }
    except requests.exceptions.RequestException as e:
        return {
            \"status\": \"error\",
            \"message\": f\"NASA ADS API request error: {str(e)}\",
            \"details\": \"Ensure ADSABS_TOKEN is valid and network is stable.\"
        }
    except Exception as e:
        return {
            \"status\": \"error\",
            \"message\": f\"An unexpected error occurred with ADS integration: {str(e)}\"
        }

if __name__ == \"__main__\":
    port = int(os.getenv(\"PORT\", 8000))
    uvicorn.run(app, host=\"0.0.0.0\", port=port, log_level=\"info\")
"'

az container create \
    --resource-group $RESOURCE_GROUP \
    --name $CONTAINER_NAME \
    --image $DOCKER_IMAGE \
    --os-type Linux \
    --cpu 1 \
    --memory 1.5 \
    --dns-name-label ${CONTAINER_NAME}-$(date +%s) \
    --ports 8000 \
    --environment-variables \
        ADSABS_TOKEN="pDbJnrgwpsZYj14Vbkn1LmMCjdiIxXrgUcrdLvjk" \
        SERPAPI_KEY="12ad8fb8fbcd01088827b7d3143de6d9fa15c571dc11ac0daf7f24b25097fac8" \
    --restart-policy Always \
    --command-line "sh -c $COMMAND_LINE_SCRIPT"

# Получаем URL
echo -e "${BLUE}Шаг 6: Получение FQDN (URL) контейнера...${NC}"
echo -e "${YELLOW}Ожидание выделения IP адреса (до 60 секунд)...${NC}"
FQDN=""
for i in {1..12}; do
    FQDN=$(az container show --resource-group $RESOURCE_GROUP --name $CONTAINER_NAME --query "ipAddress.fqdn" --output tsv 2>/dev/null)
    if [ ! -z "$FQDN" ]; then
        break
    fi
    echo -e "${YELLOW}Попытка $i/12: FQDN еще не доступен, ждем 5 секунд...${NC}"
    sleep 5
done

if [ -z "$FQDN" ]; then
    echo -e "${RED}Не удалось получить FQDN для контейнера $CONTAINER_NAME.${NC}"
    echo -e "${RED}Проверьте логи контейнера: az container logs --resource-group $RESOURCE_GROUP --name $CONTAINER_NAME${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}🎉 Free Tier API (Enhanced) успешно развернут!${NC}"
echo -e "${GREEN}Базовый URL: http://$FQDN:8000${NC}"
echo -e "${GREEN}Проверка работоспособности (ping): http://$FQDN:8000/ping${NC}"
echo -e "${GREEN}Документация API (Swagger UI): http://$FQDN:8000/docs${NC}"
echo -e "${GREEN}Пример запроса объектов: http://$FQDN:8000/astro/objects?limit=2${NC}"
echo ""

# Тестирование
echo -e "${BLUE}Шаг 7: Тестирование API (может занять до 90 секунд)...${NC}"
API_HEALTHY=false
for i in {1..18}; do
    echo -e "${YELLOW}Попытка $i/18 - проверка доступности http://$FQDN:8000/ping ...${NC}"
    if curl -fsS "http://$FQDN:8000/ping" > /tmp/ping_result.json 2>&1; then
        echo -e "${GREEN}✅ Free Tier API работает! Ответ:${NC}"
        cat /tmp/ping_result.json | jq '.' 2>/dev/null || cat /tmp/ping_result.json # Показать результат, если jq доступен
        echo ""
        API_HEALTHY=true
        break
    else
        echo -e "${YELLOW}API еще не отвечает. Код ответа curl: $?${NC}"
    fi    
    sleep 5
done

if ! $API_HEALTHY; then
    echo -e "${RED}API не ответил на /ping после 90 секунд.${NC}"
    echo -e "${RED}Проверьте логи контейнера: az container logs --resource-group $RESOURCE_GROUP --name $CONTAINER_NAME${NC}"
    echo -e "${RED}Статус контейнера:${NC}"
    az container show --resource-group $RESOURCE_GROUP --name $CONTAINER_NAME --query "instanceView.state" -o tsv
    exit 1
fi

# Тестируем другие endpoints
echo -e "${BLUE}=== Расширенное тестирование endpoints ===${NC}"

echo -e "${YELLOW}Запрос /astro/objects:${NC}"
curl -sS "http://$FQDN:8000/astro/objects?limit=2&object_type=GALAXY" | jq '.' 2>/dev/null || echo "(jq не найден для форматирования)"
echo ""

echo -e "${YELLOW}Запрос /ads/basic:${NC}"
curl -sS "http://$FQDN:8000/ads/basic?query=exoplanet&rows=1" | jq '.' 2>/dev/null || echo "(jq не найден для форматирования)"
echo ""


echo -e "${GREEN}🎯 Успешно развернуто и протестировано на Free Tier!${NC}"

echo ""
echo -e "${BLUE}=== Настройка Vercel (если используется) ===${NC}"
echo -e "${YELLOW}Добавьте в Vercel переменную окружения:${NC}"
echo -e "${GREEN}HEAVY_COMPUTE_URL=http://$FQDN:8000${NC}"

echo ""
echo -e "${BLUE}=== Характеристики Free Tier ===${NC}"
echo -e "${YELLOW}CPU:${NC} 1 core"
echo -e "${YELLOW}RAM:${NC} 1.5 GB"
echo -e "${YELLOW}Лимиты:${NC} Ограниченные наборы данных, возможны ограничения Docker Hub.${NC}"

echo ""
echo -e "${BLUE}=== Полезные команды Azure CLI ===${NC}"
echo -e "${YELLOW}Просмотр логов:${NC} az container logs --resource-group $RESOURCE_GROUP --name $CONTAINER_NAME --follow"
echo -e "${YELLOW}Статус контейнера:${NC} az container show --resource-group $RESOURCE_GROUP --name $CONTAINER_NAME --query 'instanceView.state' -o tsv"
# echo -e "${YELLOW}Перезапуск контейнера:${NC} az container restart --resource-group $RESOURCE_GROUP --name $CONTAINER_NAME"
# echo -e "${YELLOW}Остановка контейнера:${NC} az container stop --resource-group $RESOURCE_GROUP --name $CONTAINER_NAME"
echo -e "${YELLOW}Удаление контейнера:${NC} az container delete --resource-group $RESOURCE_GROUP --name $CONTAINER_NAME --yes"

# Очистка временного файла
rm -f /tmp/ping_result.json

set +x # Выключаем подробное логирование
echo -e "${GREEN}Скрипт завершен.${NC}"

exit 0 