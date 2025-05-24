#!/bin/bash

# Скрипт для проверки наличия Docker образа и его идентификатора
# Поможет правильно настроить deploy/azure-local-image.sh

set -e

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}=== Проверка локальных Docker образов ===${NC}"
echo -e "${YELLOW}Список всех локальных образов:${NC}"
docker images

echo ""
echo -e "${BLUE}=== Проверка образа по ID ===${NC}"
# Пытаемся найти образ по ID из azure-local-image.sh
IMAGE_ID="3e93fb6632cd0c1046d3319eafa871a24478f68136764ff0ca1e26815fdfd09b"
if docker inspect sha256:$IMAGE_ID &>/dev/null; then
    echo -e "${GREEN}✅ Образ с ID $IMAGE_ID найден!${NC}"
    echo -e "${YELLOW}Детали образа:${NC}"
    docker inspect sha256:$IMAGE_ID | grep -E '(RepoTags|RepoDigests|Created|Architecture|Os)'
else
    echo -e "${RED}❌ Образ с ID $IMAGE_ID не найден в локальном репозитории.${NC}"
    echo -e "${YELLOW}Список образов без тегов (dangling):${NC}"
    docker images --filter "dangling=true" -q
    
    # Проверка последнего созданного образа
    LATEST_ID=$(docker images -q | head -1)
    if [ ! -z "$LATEST_ID" ]; then
        echo -e "${YELLOW}Детали последнего созданного образа (ID: $LATEST_ID):${NC}"
        docker inspect $LATEST_ID | grep -E '(RepoTags|RepoDigests|Created|Architecture|Os)'
        echo -e "${GREEN}Для использования этого образа, обновите IMAGE_ID в deploy/azure-local-image.sh:${NC}"
        echo "LOCAL_IMAGE_ID=\"${LATEST_ID}\""
    fi
fi

echo ""
echo -e "${BLUE}=== Инструкции ===${NC}"
echo -e "1. Если образ не найден, выполните сборку образа командой:"
echo -e "   ${YELLOW}docker buildx build -t scientific-api:latest .${NC}"
echo -e "2. Получите ID образа:"
echo -e "   ${YELLOW}docker images | grep scientific-api${NC}"
echo -e "3. Обновите переменную LOCAL_IMAGE_ID в deploy/azure-local-image.sh"
echo -e "4. Запустите скрипт развертывания:"
echo -e "   ${YELLOW}./deploy/azure-local-image.sh${NC}" 