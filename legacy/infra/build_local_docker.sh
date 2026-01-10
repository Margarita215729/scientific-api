#!/bin/bash

# Скрипт для сборки Docker-образа Scientific API для Azure

set -e # Выход при ошибке

# Загрузка переменных из deploy.env
if [ -f deploy.env ]; then
    echo "Загрузка переменных из deploy.env..."
    set -a
    source deploy.env
    set +a
else
    echo "❌ Ошибка: Файл deploy.env не найден."
    exit 1
fi

# Проверка наличия DOCKER_IMAGE_FULL_PATH
if [ -z "$DOCKER_IMAGE_FULL_PATH" ]; then
    echo "❌ Ошибка: Переменная DOCKER_IMAGE_FULL_PATH не установлена в deploy.env."
    exit 1
fi

# Генерация нового тега с временной меткой
TIMESTAMP=$(date +%Y%m%d%H%M%S)
IMAGE_NAME_BASE=$(echo "$DOCKER_IMAGE_FULL_PATH" | cut -d: -f1)
NEW_IMAGE_TAG="${IMAGE_NAME_BASE}:azure-${TIMESTAMP}"

echo ""
echo "🐳 Сборка Docker-образа: $NEW_IMAGE_TAG"
echo "   Используемый BUILD_TYPE: azure"
echo ""

# Попытка 1: docker buildx build (предпочтительно, если buildx настроен)
if docker buildx build --build-arg BUILD_TYPE="azure" -t "$NEW_IMAGE_TAG" . --load; then
    echo "✅ Образ $NEW_IMAGE_TAG успешно собран с помощью 'docker buildx build'."
elif DOCKER_BUILDKIT=0 docker build --build-arg BUILD_TYPE=azure -t "$NEW_IMAGE_TAG" .; then
    # Попытка 2: DOCKER_BUILDKIT=0 docker build (классический сборщик)
    echo "✅ Образ $NEW_IMAGE_TAG успешно собран с помощью 'DOCKER_BUILDKIT=0 docker build'."
else
    echo "❌ Ошибка: Не удалось собрать Docker-образ ни одним из методов."
    exit 1
fi

echo ""
echo "🎉 Docker-образ $NEW_IMAGE_TAG готов для локального тестирования."
echo "Используйте этот тег для следующих шагов (локальный запуск, пуш в реестр)."
echo "Например, для локального запуска:"
echo ""
echo "docker run --rm -p 8000:8000 \\\\"
echo "    -e DB_TYPE=\\"$DB_TYPE\\" \\\\"
echo "    -e AZURE_COSMOS_CONNECTION_STRING=\\"$AZURE_COSMOS_CONNECTION_STRING\\" \\\\"
echo "    -e COSMOS_DATABASE_NAME=\\"$COSMOS_DATABASE_NAME\\" \\\\"
echo "    -e ADSABS_TOKEN=\\"$ADSABS_TOKEN\\" \\\\"
echo "    -e SERPAPI_KEY=\\"$SERPAPI_KEY\\" \\\\"
echo "    -e PYTHONUNBUFFERED=1 \\\\"
echo "    -e ENVIRONMENT=\\"development\\" \\\\"
echo "    -e HEAVY_PIPELINE_ON_START=\\"false\\" \\\\"
echo "    -e PORT=8000 \\\\"
echo "    --name scientific-api-local-test \\\\"
echo "    \\\"$NEW_IMAGE_TAG\\\""
echo ""