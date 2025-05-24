#!/bin/bash

# Скрипт для проверки наличия Docker образов
# Помогает диагностировать проблемы перед развертыванием

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}=== Проверка Docker образов для развертывания ===${NC}"

# Проверка, установлен ли Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Docker не установлен или не найден в системном пути.${NC}"
    echo -e "${YELLOW}Установите Docker перед развертыванием: https://docs.docker.com/get-docker/${NC}"
    exit 1
fi

# Проверка статуса Docker
echo -e "${BLUE}Проверка статуса Docker...${NC}"
if ! docker info &> /dev/null; then
    echo -e "${RED}Docker не запущен или требует привилегии sudo.${NC}"
    echo -e "${YELLOW}Запустите Docker или используйте sudo для выполнения команд Docker.${NC}"
    exit 1
fi
echo -e "${GREEN}Docker запущен и работает корректно.${NC}"

# Вывод информации о платформе
echo -e "${BLUE}Информация о системе:${NC}"
echo -e "${YELLOW}Архитектура CPU:${NC} $(uname -m)"
echo -e "${YELLOW}Операционная система:${NC} $(uname -s)"
echo -e "${YELLOW}Docker платформы:${NC} $(docker info | grep "Server OS/Arch" | sed 's/Server OS\/Arch: *//')"

# Проверка наличия образов scientific-api
echo -e "\n${BLUE}Проверка локальных образов scientific-api...${NC}"
if ! docker images | grep -q "scientific-api"; then
    echo -e "${RED}Образы scientific-api не найдены.${NC}"
    echo -e "${YELLOW}Соберите образ перед развертыванием:${NC}"
    echo -e "${GREEN}docker build -t scientific-api:latest .${NC}"
else
    echo -e "${GREEN}Найдены следующие образы scientific-api:${NC}"
    docker images | grep "scientific-api" | awk '{printf "  %s:%s (ID: %s, Создан: %s)\n", $1, $2, $3, $4" "$5" "$6}'
fi

# Проверка доступного места для Docker
echo -e "\n${BLUE}Проверка доступного места для Docker...${NC}"
if command -v docker &> /dev/null && command -v df &> /dev/null; then
    # Определение папки с Docker данными
    DOCKER_ROOT=$(docker info | grep "Docker Root Dir" | cut -d: -f2 | tr -d ' ')
    if [ -z "$DOCKER_ROOT" ]; then
        DOCKER_ROOT="/var/lib/docker"  # По умолчанию для большинства систем
    fi
    
    # Получение информации о доступном месте
    DF_OUTPUT=$(df -h "$DOCKER_ROOT" 2>/dev/null || df -h / 2>/dev/null)
    if [ -n "$DF_OUTPUT" ]; then
        echo -e "${YELLOW}Информация о диске:${NC}"
        echo "$DF_OUTPUT" | head -1
        echo "$DF_OUTPUT" | grep -v "Filesystem" | head -1
    else
        echo -e "${YELLOW}Не удалось получить информацию о диске.${NC}"
    fi
fi

# Информация о buildx
echo -e "\n${BLUE}Проверка наличия Docker Buildx (для мульти-архитектурных образов)...${NC}"
if docker buildx version &> /dev/null; then
    echo -e "${GREEN}Docker Buildx установлен:${NC}"
    docker buildx version
    
    echo -e "\n${YELLOW}Доступные платформы для сборки:${NC}"
    docker buildx inspect | grep "Platforms" || echo "  Не удалось получить список платформ"
else
    echo -e "${YELLOW}Docker Buildx не установлен или недоступен.${NC}"
    echo -e "${YELLOW}Для сборки мульти-архитектурных образов рекомендуется установить buildx:${NC}"
    echo -e "${GREEN}https://docs.docker.com/buildx/working-with-buildx/${NC}"
fi

# Проверка Azure CLI
echo -e "\n${BLUE}Проверка наличия Azure CLI...${NC}"
if command -v az &> /dev/null; then
    echo -e "${GREEN}Azure CLI установлен:${NC}"
    az --version | head -1
    
    # Проверка авторизации в Azure
    echo -e "\n${YELLOW}Проверка авторизации в Azure...${NC}"
    if az account show &> /dev/null; then
        echo -e "${GREEN}Авторизация в Azure выполнена:${NC}"
        az account show --query name -o tsv
    else
        echo -e "${RED}Авторизация в Azure не выполнена.${NC}"
        echo -e "${YELLOW}Выполните вход в Azure:${NC}"
        echo -e "${GREEN}az login${NC}"
    fi
else
    echo -e "${YELLOW}Azure CLI не установлен или недоступен.${NC}"
    echo -e "${YELLOW}Для развертывания в Azure установите Azure CLI:${NC}"
    echo -e "${GREEN}https://docs.microsoft.com/ru-ru/cli/azure/install-azure-cli${NC}"
fi

echo -e "\n${BLUE}=== Инструкции по развертыванию ===${NC}"
echo -e "${YELLOW}1. Для развертывания с использованием локального образа arm64:${NC}"
echo -e "${GREEN}   ./deploy/azure-local-image.sh${NC}"
echo -e "${YELLOW}2. Для сборки и развертывания AMD64-совместимого образа:${NC}"
echo -e "${GREEN}   ./deploy/azure-amd64.sh${NC}"

echo -e "\n${GREEN}Проверка завершена.${NC}"
exit 0 