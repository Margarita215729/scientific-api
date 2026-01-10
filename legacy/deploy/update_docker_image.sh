#!/bin/bash

# Скрипт для обновления Docker образа в деплое
# Используйте этот скрипт, если хотите переключиться на аккаунт cutypie

echo "🔄 Обновление Docker образа в деплое"
echo "====================================="

# Проверяем, какой образ использовать
if [ "$1" = "cutypie" ]; then
    NEW_IMAGE="cutypie/scientific-api-app-image:scientific-api"
    echo "📦 Используем образ: $NEW_IMAGE"
elif [ "$1" = "gretk" ]; then
    NEW_IMAGE="gretk/scientific-api-app-image:scientific-api"
    echo "📦 Используем образ: $NEW_IMAGE"
else
    echo "❌ Укажите аккаунт: ./update_docker_image.sh [cutypie|gretk]"
    exit 1
fi

# Обновляем deploy_production_final.sh
echo "📝 Обновляем deploy_production_final.sh..."
sed -i '' "s|gretk/scientific-api-app-image:scientific-api|$NEW_IMAGE|g" deploy_production_final.sh

# Обновляем update_azure_container.sh
echo "📝 Обновляем update_azure_container.sh..."
sed -i '' "s|gretk/scientific-api-app-image:scientific-api|$NEW_IMAGE|g" update_azure_container.sh

echo "✅ Обновление завершено!"
echo "🚀 Теперь можно запустить деплой: ./deploy_production_final.sh"
