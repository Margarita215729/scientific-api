#!/bin/bash

# Упрощенный скрипт развертывания для начинающих с Azure
# Пошаговые инструкции и проверки

set -e

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Конфигурация (ваши данные)
RESOURCE_GROUP="scientific-api"
SUBSCRIPTION_ID="8e746503-c0c0-4535-a05d-49e544196e3f"
CONTAINER_NAME="scientific-api-heavy"
LOCATION="eastus"
REGISTRY_NAME="scientificapi$(date +%s)"

echo -e "${GREEN}=== Развертывание Scientific API в Azure ===${NC}"
echo -e "${BLUE}Resource Group: $RESOURCE_GROUP${NC}"
echo -e "${BLUE}Subscription: $SUBSCRIPTION_ID${NC}"
echo ""

# Функция для ожидания нажатия Enter
wait_for_enter() {
    echo -e "${YELLOW}Нажмите Enter для продолжения...${NC}"
    read
}

# Проверка 1: Azure CLI
echo -e "${YELLOW}Шаг 1: Проверка Azure CLI...${NC}"
if ! command -v az &> /dev/null; then
    echo -e "${RED}❌ Azure CLI не установлен${NC}"
    echo -e "${YELLOW}Устанавливаем Azure CLI...${NC}"
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        brew install azure-cli
    else
        # Linux
        curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
    fi
else
    echo -e "${GREEN}✅ Azure CLI установлен${NC}"
fi

# Проверка 2: Docker
echo -e "${YELLOW}Шаг 2: Проверка Docker...${NC}"
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker не установлен${NC}"
    echo -e "${YELLOW}Пожалуйста, установите Docker Desktop с официального сайта${NC}"
    echo -e "${BLUE}https://www.docker.com/products/docker-desktop${NC}"
    exit 1
fi

if ! docker info &> /dev/null; then
    echo -e "${RED}❌ Docker не запущен${NC}"
    echo -e "${YELLOW}Пожалуйста, запустите Docker Desktop${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Docker работает${NC}"

# Проверка 3: Авторизация в Azure
echo -e "${YELLOW}Шаг 3: Проверка авторизации в Azure...${NC}"
if ! az account show &> /dev/null; then
    echo -e "${RED}❌ Не авторизованы в Azure${NC}"
    echo -e "${YELLOW}Выполняем авторизацию...${NC}"
    az login
else
    echo -e "${GREEN}✅ Авторизация в Azure выполнена${NC}"
fi

# Установка правильной подписки
echo -e "${YELLOW}Устанавливаем подписку...${NC}"
az account set --subscription $SUBSCRIPTION_ID
echo -e "${GREEN}✅ Подписка установлена${NC}"

# Проверка 4: Resource Group
echo -e "${YELLOW}Шаг 4: Проверка Resource Group...${NC}"
if az group show --name $RESOURCE_GROUP &> /dev/null; then
    echo -e "${GREEN}✅ Resource Group '$RESOURCE_GROUP' найдена${NC}"
else
    echo -e "${YELLOW}Создаем Resource Group...${NC}"
    az group create --name $RESOURCE_GROUP --location $LOCATION
    echo -e "${GREEN}✅ Resource Group создана${NC}"
fi

echo ""
echo -e "${BLUE}=== Готовы к развертыванию ===${NC}"
wait_for_enter

# Создание Container Registry
echo -e "${YELLOW}Шаг 5: Создание Container Registry...${NC}"
if az acr show --name $REGISTRY_NAME --resource-group $RESOURCE_GROUP &> /dev/null; then
    echo -e "${GREEN}✅ Registry уже существует${NC}"
else
    echo -e "${YELLOW}Создаем Container Registry (это займет 2-3 минуты)...${NC}"
    az acr create --resource-group $RESOURCE_GROUP --name $REGISTRY_NAME --sku Basic --admin-enabled true
    echo -e "${GREEN}✅ Container Registry создан${NC}"
fi

# Получение данных для входа
ACR_LOGIN_SERVER=$(az acr show --name $REGISTRY_NAME --resource-group $RESOURCE_GROUP --query "loginServer" --output tsv)
ACR_USERNAME=$(az acr credential show --name $REGISTRY_NAME --query "username" --output tsv)
ACR_PASSWORD=$(az acr credential show --name $REGISTRY_NAME --query "passwords[0].value" --output tsv)

echo -e "${GREEN}✅ Registry: $ACR_LOGIN_SERVER${NC}"

# Сборка Docker образа
echo -e "${YELLOW}Шаг 6: Сборка Docker образа...${NC}"
echo -e "${YELLOW}Это займет 5-10 минут...${NC}"
docker build -f docker/heavy-compute.dockerfile -t scientific-api-heavy:latest .

# Тегирование для Registry
echo -e "${YELLOW}Подготовка образа для загрузки...${NC}"
docker tag scientific-api-heavy:latest $ACR_LOGIN_SERVER/scientific-api-heavy:latest

# Вход в Registry
echo -e "${YELLOW}Авторизация в Container Registry...${NC}"
az acr login --name $REGISTRY_NAME

# Загрузка образа
echo -e "${YELLOW}Загрузка образа (это займет 5-10 минут)...${NC}"
docker push $ACR_LOGIN_SERVER/scientific-api-heavy:latest
echo -e "${GREEN}✅ Образ загружен${NC}"

# Проверка переменных окружения
echo -e "${YELLOW}Шаг 7: Проверка переменных окружения...${NC}"
if [[ -z "$ADSABS_TOKEN" ]]; then
    echo -e "${YELLOW}⚠️ Переменная ADSABS_TOKEN не установлена${NC}"
    echo -e "${YELLOW}Установите её: export ADSABS_TOKEN='your_token'${NC}"
fi

if [[ -z "$SERPAPI_KEY" ]]; then
    echo -e "${YELLOW}⚠️ Переменная SERPAPI_KEY не установлена${NC}"
fi

# Развертывание контейнера
echo -e "${YELLOW}Шаг 8: Развертывание контейнера...${NC}"
echo -e "${YELLOW}Создаем контейнер с 12 CPU и 20GB RAM...${NC}"

az container create \
    --resource-group $RESOURCE_GROUP \
    --name $CONTAINER_NAME \
    --image $ACR_LOGIN_SERVER/scientific-api-heavy:latest \
    --registry-login-server $ACR_LOGIN_SERVER \
    --registry-username $ACR_USERNAME \
    --registry-password $ACR_PASSWORD \
    --cpu 12 \
    --memory 20 \
    --dns-name-label scientific-api-heavy-$(date +%s) \
    --ports 8000 \
    --environment-variables \
        ENVIRONMENT=production \
        ADSABS_TOKEN="$ADSABS_TOKEN" \
        SERPAPI_KEY="$SERPAPI_KEY" \
        GOOGLE_CLIENT_ID="$GOOGLE_CLIENT_ID" \
        GOOGLE_CLIENT_SECRET="$GOOGLE_CLIENT_SECRET" \
        GOOGLE_REFRESH_TOKEN="$GOOGLE_REFRESH_TOKEN" \
    --restart-policy Always

# Получение URL
FQDN=$(az container show --resource-group $RESOURCE_GROUP --name $CONTAINER_NAME --query "ipAddress.fqdn" --output tsv)

echo ""
echo -e "${GREEN}🎉 Развертывание завершено!${NC}"
echo -e "${GREEN}URL вашего API: http://$FQDN:8000${NC}"
echo -e "${GREEN}Health check: http://$FQDN:8000/ping${NC}"
echo ""

# Тестирование
echo -e "${YELLOW}Ожидаем запуска контейнера (30 секунд)...${NC}"
sleep 30

echo -e "${YELLOW}Тестируем развертывание...${NC}"
if curl -f "http://$FQDN:8000/ping" > /dev/null 2>&1; then
    echo -e "${GREEN}✅ API работает!${NC}"
    echo ""
    echo -e "${BLUE}=== Полезные команды ===${NC}"
    echo -e "${YELLOW}Просмотр логов:${NC}"
    echo "az container logs --resource-group $RESOURCE_GROUP --name $CONTAINER_NAME"
    echo ""
    echo -e "${YELLOW}Статус контейнера:${NC}"
    echo "az container show --resource-group $RESOURCE_GROUP --name $CONTAINER_NAME"
    echo ""
    echo -e "${YELLOW}Для Vercel добавьте переменную:${NC}"
    echo "HEAVY_COMPUTE_URL=http://$FQDN:8000"
else
    echo -e "${RED}❌ API не отвечает${NC}"
    echo -e "${YELLOW}Проверьте логи:${NC}"
    echo "az container logs --resource-group $RESOURCE_GROUP --name $CONTAINER_NAME"
fi 