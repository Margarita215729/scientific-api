# 🚀 Production Deployment - Scientific API

## Полнофункциональный деплой в Azure Container Instances

**БЕЗ УПРОЩЕНИЙ! ПОЛНАЯ PRODUCTION ВЕРСИЯ!**

### 🎯 Что развертывается

- ✅ **Все астрономические каталоги**: SDSS DR17, Euclid Q1, DESI DR1, DES Y6
- ✅ **Полная обработка данных**: удаление дубликатов, ML-ready features
- ✅ **NASA ADS интеграция**: поиск научной литературы
- ✅ **Современный веб-интерфейс**: Bootstrap, интерактивные графики
- ✅ **Все библиотеки**: astropy, pandas, numpy, scikit-learn, matplotlib
- ✅ **4 CPU cores, 8GB RAM**: мощные ресурсы для обработки данных

## 🛠️ Предварительные требования

### 1. Установите Azure CLI
```bash
# macOS
brew install azure-cli

# Windows
winget install Microsoft.AzureCLI

# Linux
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
```

### 2. Установите Docker
```bash
# macOS
brew install docker

# Или скачайте Docker Desktop с https://docker.com
```

### 3. Войдите в Azure
```bash
az login
```

## 🚀 Быстрый деплой

### Вариант 1: Автоматический деплой (рекомендуется)

```bash
# Сделайте скрипт исполняемым
chmod +x deploy_azure_production.sh

# Запустите деплой
./deploy_azure_production.sh
```

### Вариант 2: Пошаговый деплой

```bash
# 1. Создайте ресурсную группу
az group create --name scientific-api-production --location eastus

# 2. Создайте Container Registry
az acr create \
    --resource-group scientific-api-production \
    --name scientificapiregistry \
    --sku Basic \
    --admin-enabled true

# 3. Соберите Docker образ
docker build -t scientific-api:production .

# 4. Получите данные реестра
ACR_LOGIN_SERVER=$(az acr show --name scientificapiregistry --resource-group scientific-api-production --query "loginServer" --output tsv)

# 5. Тегируйте образ
docker tag scientific-api:production $ACR_LOGIN_SERVER/scientific-api:production

# 6. Войдите в реестр
az acr login --name scientificapiregistry

# 7. Загрузите образ
docker push $ACR_LOGIN_SERVER/scientific-api:production

# 8. Получите учетные данные
ACR_USERNAME=$(az acr credential show --name scientificapiregistry --resource-group scientific-api-production --query "username" --output tsv)
ACR_PASSWORD=$(az acr credential show --name scientificapiregistry --resource-group scientific-api-production --query "passwords[0].value" --output tsv)

# 9. Создайте контейнер
az container create \
    --resource-group scientific-api-production \
    --name scientific-api-prod \
    --image $ACR_LOGIN_SERVER/scientific-api:production \
    --registry-login-server $ACR_LOGIN_SERVER \
    --registry-username $ACR_USERNAME \
    --registry-password $ACR_PASSWORD \
    --dns-name-label scientific-api-prod \
    --ports 8000 \
    --cpu 4 \
    --memory 8 \
    --environment-variables ENVIRONMENT=production PYTHONPATH=/app \
    --restart-policy Always
```

## 🌐 После деплоя

Ваше приложение будет доступно по адресу:
```
http://scientific-api-prod.eastus.azurecontainer.io:8000
```

### 📚 Основные эндпоинты:

- **Главная страница**: `/`
- **API документация**: `/api/docs`
- **Health check**: `/api/health`
- **ADS поиск**: `/ads`

### 🔧 Астрономические данные:

- **Статус каталогов**: `/api/astro/status`
- **Загрузка данных**: `POST /api/astro/download`
- **Получение галактик**: `/api/astro/galaxies`
- **Статистика**: `/api/astro/statistics`

### 📖 NASA ADS:

- **Поиск по координатам**: `/api/ads/search-by-coordinates`
- **Поиск по объекту**: `/api/ads/search-by-object`
- **Поиск LSS**: `/api/ads/large-scale-structure`

## 🔧 Управление контейнером

### Просмотр логов
```bash
az container logs --resource-group scientific-api-production --name scientific-api-prod
```

### Перезапуск
```bash
az container restart --resource-group scientific-api-production --name scientific-api-prod
```

### Получение статуса
```bash
az container show --resource-group scientific-api-production --name scientific-api-prod
```

### Удаление
```bash
az container delete --resource-group scientific-api-production --name scientific-api-prod --yes
```

## 🧪 Локальное тестирование

Перед деплоем в Azure можете протестировать локально:

```bash
# Сборка и запуск
docker-compose up --build

# Приложение будет доступно на http://localhost:8000
```

## 📊 Мониторинг

### Health Check
```bash
curl http://scientific-api-prod.eastus.azurecontainer.io:8000/api/health
```

### Проверка загрузки данных
```bash
curl http://scientific-api-prod.eastus.azurecontainer.io:8000/api/astro/status
```

## 🔒 Безопасность

### Переменные окружения (опционально)
Добавьте в деплой:
```bash
--environment-variables \
    ADSABS_TOKEN=your_ads_token \
    ENVIRONMENT=production \
    PYTHONPATH=/app
```

## 💰 Стоимость

Azure Container Instances:
- **4 vCPU, 8GB RAM**: ~$120/месяц
- **Container Registry**: ~$5/месяц
- **Трафик**: зависит от использования

## 🚨 Важные заметки

1. **Первый запуск**: Контейнер загрузит и обработает астрономические данные (~10-15 минут)
2. **Память**: 8GB достаточно для обработки всех каталогов
3. **Хранение**: Данные сохраняются в контейнере (эфемерное хранилище)
4. **Автозапуск**: Контейнер автоматически перезапускается при сбоях

## 🎉 Готово!

Ваш полнофункциональный Scientific API развернут в production без каких-либо упрощений!

**Все функции работают:**
- ✅ Реальные астрономические каталоги
- ✅ Обработка FITS файлов
- ✅ ML-ready features
- ✅ NASA ADS интеграция
- ✅ Современный веб-интерфейс
- ✅ Полная документация API 