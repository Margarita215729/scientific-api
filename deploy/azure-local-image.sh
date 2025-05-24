#!/bin/bash

# Развертывание в Azure используя локальный Docker образ
# Решает проблему с ограничениями Docker Hub
# Использует минимальные ресурсы: 1 CPU, 1.5GB RAM

set -e  # Выход при любой ошибке

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Конфигурация
RESOURCE_GROUP="scientific-api"
SUBSCRIPTION_ID="8e746503-c0c0-4535-a05d-49e544196e3f"
CONTAINER_NAME="scientific-api-local"
LOCATION="eastus"

# Локальный образ и ACR конфигурация
LOCAL_IMAGE_TAG="scientific-api:latest"                          # Тег для локального образа
ACR_NAME="scientificapiacr"                                      # Имя Azure Container Registry
ACR_IMAGE="${ACR_NAME}.azurecr.io/scientific-api:latest"         # Полный путь образа в ACR

echo -e "${GREEN}=== Развертывание Scientific API с локальным образом Docker ===${NC}"
echo -e "${YELLOW}Используем ресурсы: 1 CPU, 1.5GB RAM в регионе $LOCATION${NC}"

# Проверка наличия локального образа
echo -e "${BLUE}Проверка наличия локального образа...${NC}"
if ! docker images | grep -q "$LOCAL_IMAGE_TAG"; then
    echo -e "${RED}Образ $LOCAL_IMAGE_TAG не найден. Пожалуйста, соберите образ перед запуском:${NC}"
    echo -e "${YELLOW}docker build -t $LOCAL_IMAGE_TAG .${NC}"
    exit 1
fi

# Получение ID локального образа
LOCAL_IMAGE_ID=$(docker images $LOCAL_IMAGE_TAG --quiet)
if [ -z "$LOCAL_IMAGE_ID" ]; then
    echo -e "${RED}Не удалось получить ID образа $LOCAL_IMAGE_TAG${NC}"
    exit 1
fi
echo -e "${GREEN}Найден образ с ID: $LOCAL_IMAGE_ID${NC}"

# Определение архитектуры образа
IMAGE_ARCH=$(docker inspect --format '{{.Architecture}}' $LOCAL_IMAGE_ID)
echo -e "${YELLOW}Архитектура образа: $IMAGE_ARCH${NC}"

# Шаг 1: Установка активной подписки
echo -e "${BLUE}Шаг 1: Установка активной подписки Azure...${NC}"
az account set --subscription $SUBSCRIPTION_ID
echo -e "${GREEN}Подписка установлена: $SUBSCRIPTION_ID${NC}"

# Шаг 2: Проверка и создание группы ресурсов
echo -e "${BLUE}Шаг 2: Проверка/создание группы ресурсов...${NC}"
if az group show --name $RESOURCE_GROUP &>/dev/null; then
    echo -e "${GREEN}Группа ресурсов $RESOURCE_GROUP уже существует.${NC}"
else
    echo -e "${YELLOW}Создание группы ресурсов $RESOURCE_GROUP в $LOCATION...${NC}"
    az group create --name $RESOURCE_GROUP --location $LOCATION
    echo -e "${GREEN}Группа ресурсов $RESOURCE_GROUP создана.${NC}"
fi

# Шаг 3: Очистка старых контейнеров
echo -e "${BLUE}Шаг 3: Очистка старого контейнера $CONTAINER_NAME...${NC}"
if az container show --resource-group $RESOURCE_GROUP --name $CONTAINER_NAME --query name -o tsv 2>/dev/null; then
    echo -e "${YELLOW}Удаляем существующий контейнер $CONTAINER_NAME...${NC}"
    az container delete --resource-group $RESOURCE_GROUP --name $CONTAINER_NAME --yes
    echo -e "${GREEN}Ожидание завершения удаления (30 сек)...${NC}"
    sleep 30
else
    echo -e "${GREEN}Контейнер $CONTAINER_NAME не найден. Очистка не требуется.${NC}"
fi

# Шаг 4: Создание Azure Container Registry (если не существует)
echo -e "${BLUE}Шаг 4: Проверка/создание Azure Container Registry...${NC}"
if az acr show --name $ACR_NAME --resource-group $RESOURCE_GROUP &>/dev/null; then
    echo -e "${GREEN}Azure Container Registry $ACR_NAME уже существует.${NC}"
else
    echo -e "${YELLOW}Создание Azure Container Registry $ACR_NAME...${NC}"
    az acr create --resource-group $RESOURCE_GROUP --name $ACR_NAME --sku Basic --admin-enabled true
    echo -e "${GREEN}Azure Container Registry $ACR_NAME создан.${NC}"
fi

# Шаг 5: Получение учетных данных ACR
echo -e "${BLUE}Шаг 5: Получение учетных данных ACR...${NC}"
ACR_USERNAME=$(az acr credential show --name $ACR_NAME --query "username" -o tsv)
ACR_PASSWORD=$(az acr credential show --name $ACR_NAME --query "passwords[0].value" -o tsv)
echo -e "${GREEN}Учетные данные ACR получены.${NC}"

# Шаг 6: Пометка локального образа
echo -e "${BLUE}Шаг 6: Пометка локального образа для отправки в ACR...${NC}"
docker tag $LOCAL_IMAGE_ID $ACR_IMAGE
echo -e "${GREEN}Образ помечен как $ACR_IMAGE${NC}"

# Шаг 7: Вход в ACR
echo -e "${BLUE}Шаг 7: Вход в Azure Container Registry...${NC}"
echo $ACR_PASSWORD | docker login $ACR_NAME.azurecr.io --username $ACR_USERNAME --password-stdin
echo -e "${GREEN}Вход в ACR выполнен успешно.${NC}"

# Шаг 8: Отправка образа в ACR
echo -e "${BLUE}Шаг 8: Отправка образа в ACR (это может занять несколько минут)...${NC}"
docker push $ACR_IMAGE
echo -e "${GREEN}Образ успешно отправлен в ACR.${NC}"

# Определение типа ОС на основе архитектуры
OS_TYPE="Linux"

# Шаг 9: Создание контейнера в Azure
echo -e "${BLUE}Шаг 9: Создание контейнера в Azure из образа в ACR...${NC}"
az container create \
    --resource-group $RESOURCE_GROUP \
    --name $CONTAINER_NAME \
    --image $ACR_IMAGE \
    --os-type $OS_TYPE \
    --cpu 1 \
    --memory 1.5 \
    --registry-login-server $ACR_NAME.azurecr.io \
    --registry-username $ACR_USERNAME \
    --registry-password $ACR_PASSWORD \
    --dns-name-label scientific-api-local-$(date +%s) \
    --ports 8000 \
    --environment-variables \
        ADSABS_TOKEN="pDbJnrgwpsZYj14Vbkn1LmMCjdiIxXrgUcrdLvjk" \
        SERPAPI_KEY="12ad8fb8fbcd01088827b7d3143de6d9fa15c571dc11ac0daf7f24b25097fac8" \
    --restart-policy Always

# Шаг 10: Получение FQDN
echo -e "${BLUE}Шаг 10: Получение URL контейнера...${NC}"
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
echo -e "${GREEN}🎉 Scientific API успешно развернут с использованием локального образа!${NC}"
echo -e "${GREEN}Базовый URL: http://$FQDN:8000${NC}"
echo -e "${GREEN}Документация API: http://$FQDN:8000/docs${NC}"
echo ""

# Шаг 11: Тестирование API
echo -e "${BLUE}Шаг 11: Тестирование API (может занять до 120 секунд)...${NC}"
API_HEALTHY=false
for i in {1..24}; do
    echo -e "${YELLOW}Попытка $i/24 - проверка доступности API...${NC}"
    if curl -fsS "http://$FQDN:8000/ping" > /tmp/ping_result.json 2>&1; then
        echo -e "${GREEN}✅ API работает! Ответ:${NC}"
        cat /tmp/ping_result.json | jq '.' 2>/dev/null || cat /tmp/ping_result.json
        echo ""
        API_HEALTHY=true
        break
    else
        echo -e "${YELLOW}API еще не отвечает. Код ответа curl: $?${NC}"
    fi    
    sleep 5
done

if ! $API_HEALTHY; then
    echo -e "${RED}API не ответил после нескольких попыток.${NC}"
    echo -e "${RED}Просмотр логов контейнера:${NC}"
    az container logs --resource-group $RESOURCE_GROUP --name $CONTAINER_NAME
    echo -e "${RED}Статус контейнера:${NC}"
    az container show --resource-group $RESOURCE_GROUP --name $CONTAINER_NAME --query "instanceView.state" -o tsv
    exit 1
fi

echo -e "${GREEN}🎯 Успешно развернуто и протестировано в Azure!${NC}"
echo ""
echo -e "${BLUE}=== Настройка Vercel (если используется) ===${NC}"
echo -e "${YELLOW}Добавьте в Vercel переменную окружения:${NC}"
echo -e "${GREEN}HEAVY_COMPUTE_URL=http://$FQDN:8000${NC}"

echo ""
echo -e "${BLUE}=== Полезные команды ===${NC}"
echo -e "${YELLOW}Просмотр логов:${NC} az container logs --resource-group $RESOURCE_GROUP --name $CONTAINER_NAME --follow"
echo -e "${YELLOW}Статус контейнера:${NC} az container show --resource-group $RESOURCE_GROUP --name $CONTAINER_NAME --query 'instanceView.state'"
echo -e "${YELLOW}Удалить контейнер:${NC} az container delete --resource-group $RESOURCE_GROUP --name $CONTAINER_NAME --yes"
echo -e "${YELLOW}Удалить образ из ACR:${NC} az acr repository delete --name $ACR_NAME --image scientific-api:latest --yes"

# Очистка временного файла
rm -f /tmp/ping_result.json

echo -e "${GREEN}Скрипт завершен.${NC}"
exit 0 