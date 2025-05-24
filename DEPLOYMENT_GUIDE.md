# Scientific API - Deployment Guide

## Архитектура системы

Система состоит из двух компонентов:

1. **Light API** (Vercel) - Легкий API для базовых операций
2. **Heavy Compute Service** (Azure Container Instances) - Тяжелые вычисления с реальными данными

## Предварительные требования

### API Ключи
Убедитесь, что у вас есть следующие API ключи в `.env`:

```bash
# NASA ADS API
ADSABS_TOKEN=your_ads_token_here

# SerpAPI для поиска
SERPAPI_KEY=your_serpapi_key_here

# Google API для Drive интеграции
GOOGLE_CLIENT_ID=your_google_client_id
GOOGLE_CLIENT_SECRET=your_google_client_secret
GOOGLE_REFRESH_TOKEN=your_google_refresh_token
```

### Инструменты
- Azure CLI
- Docker
- Node.js (для Vercel)
- Python 3.11+

## Развертывание Heavy Compute Service на Azure

### 1. Подготовка

```bash
# Клонируйте репозиторий
git clone <your-repo>
cd scientific-api

# Установите Azure CLI
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

# Войдите в Azure
az login
```

### 2. Настройка переменных окружения

```bash
# Экспортируйте переменные для Azure
export ADSABS_TOKEN="your_ads_token"
export SERPAPI_KEY="your_serpapi_key"
export GOOGLE_CLIENT_ID="your_google_client_id"
export GOOGLE_CLIENT_SECRET="your_google_client_secret"
export GOOGLE_REFRESH_TOKEN="your_google_refresh_token"
```

### 3. Развертывание

```bash
# Сделайте скрипт исполняемым
chmod +x deploy/azure-deploy.sh

# Запустите развертывание
./deploy/azure-deploy.sh
```

Скрипт автоматически:
- Создаст Resource Group
- Создаст Azure Container Registry
- Соберет и загрузит Docker образ
- Развернет контейнер с 12 CPU и 20GB RAM
- Настроит DNS и переменные окружения

### 4. Проверка развертывания

После успешного развертывания вы получите URL:
```
Heavy Compute Service URL: http://scientific-api-heavy.eastus.azurecontainer.io:8000
```

Проверьте работоспособность:
```bash
curl http://scientific-api-heavy.eastus.azurecontainer.io:8000/ping
```

## Развертывание Light API на Vercel

### 1. Подготовка Vercel проекта

```bash
# Установите Vercel CLI
npm i -g vercel

# Войдите в Vercel
vercel login
```

### 2. Настройка переменных окружения в Vercel

В панели Vercel добавьте переменную:
```
HEAVY_COMPUTE_URL=http://scientific-api-heavy.eastus.azurecontainer.io:8000
```

### 3. Развертывание

```bash
# Разверните на Vercel
vercel --prod
```

## Функциональность системы

### Основные возможности

1. **Загрузка реальных астрономических данных**:
   - SDSS DR17 spectroscopic catalog
   - Euclid Q1 MER Final catalog
   - DESI DR1 ELG clustering catalog
   - DES Y6 Gold catalog

2. **Обработка данных**:
   - Очистка и нормализация
   - Feature engineering
   - Подготовка ML-ready датасетов

3. **Поиск в NASA ADS**:
   - По координатам
   - По названию объекта
   - По каталогу
   - По крупномасштабным структурам

4. **ML Pipeline**:
   - Автоматическая подготовка датасетов
   - Нормализация и масштабирование
   - Train/test split
   - Экспорт в ZIP архивы

### API Endpoints

#### Light API (Vercel)
- `GET /ping` - Health check
- `GET /astro/status` - Статус каталогов
- `GET /astro/statistics` - Статистика данных
- `GET /astro/galaxies` - Данные галактик с фильтрацией
- `GET /ads/*` - Поиск в ADS (демо данные)

#### Heavy Compute API (Azure)
- `POST /astro/download` - Загрузка реальных данных
- `GET /astro/download/{task_id}` - Статус загрузки
- `POST /ml/prepare-dataset` - Подготовка ML датасета
- `GET /ml/dataset/{task_id}/download` - Скачивание датасета
- `POST /data/process` - Кастомная обработка данных

## Использование системы

### 1. Загрузка астрономических данных

```bash
# Запустите загрузку данных
curl -X POST http://your-heavy-service/astro/download

# Получите task_id и проверяйте статус
curl http://your-heavy-service/astro/download/{task_id}
```

### 2. Подготовка ML датасета

```bash
# Подготовьте датасет для предсказания redshift
curl -X POST "http://your-heavy-service/ml/prepare-dataset?target_variable=redshift&test_size=0.2&normalization=standard"

# Скачайте готовый датасет
curl -O http://your-heavy-service/ml/dataset/{task_id}/download
```

### 3. Поиск в NASA ADS

```bash
# Поиск по координатам
curl "http://your-heavy-service/ads/search?search_type=coordinates&ra=150.0&dec=2.0&radius=0.1"

# Поиск по объекту
curl "http://your-heavy-service/ads/search?search_type=object&query=M31"
```

## Мониторинг и логи

### Azure Container Instances

```bash
# Просмотр логов
az container logs --resource-group scientific-api-rg --name scientific-api-heavy

# Статус контейнера
az container show --resource-group scientific-api-rg --name scientific-api-heavy

# Перезапуск контейнера
az container restart --resource-group scientific-api-rg --name scientific-api-heavy
```

### Vercel

Логи доступны в панели Vercel или через CLI:
```bash
vercel logs
```

## Масштабирование

### Увеличение ресурсов Azure

Для обработки больших датасетов можно увеличить ресурсы:

```bash
# Обновите контейнер с большими ресурсами
az container create \
    --resource-group scientific-api-rg \
    --name scientific-api-heavy-large \
    --cpu 16 \
    --memory 32 \
    # ... остальные параметры
```

### Оптимизация производительности

1. **Кэширование**: Данные кэшируются локально
2. **Chunked processing**: Большие файлы обрабатываются по частям
3. **Background tasks**: Длительные операции выполняются асинхронно
4. **Fallback**: Light API работает даже без Heavy Service

## Безопасность

1. **API ключи**: Хранятся в переменных окружения
2. **HTTPS**: Используйте HTTPS в продакшене
3. **Rate limiting**: Настройте ограничения запросов
4. **Мониторинг**: Отслеживайте использование ресурсов

## Troubleshooting

### Частые проблемы

1. **Контейнер не запускается**:
   ```bash
   az container logs --resource-group scientific-api-rg --name scientific-api-heavy
   ```

2. **Нет доступа к данным**:
   - Проверьте API ключи
   - Убедитесь в доступности внешних сервисов

3. **Медленная обработка**:
   - Увеличьте ресурсы контейнера
   - Проверьте сетевое соединение

4. **Ошибки памяти**:
   - Увеличьте память контейнера
   - Уменьшите размер обрабатываемых данных

### Полезные команды

```bash
# Проверка статуса всех ресурсов
az resource list --resource-group scientific-api-rg --output table

# Удаление всех ресурсов
az group delete --name scientific-api-rg --yes

# Просмотр метрик контейнера
az monitor metrics list --resource /subscriptions/{subscription}/resourceGroups/scientific-api-rg/providers/Microsoft.ContainerInstance/containerGroups/scientific-api-heavy
```

## Стоимость

Примерная стоимость Azure Container Instances:
- 12 vCPU, 20GB RAM: ~$200-300/месяц при постоянной работе
- Рекомендуется настроить автоматическое отключение при неактивности

## Поддержка

При возникновении проблем:
1. Проверьте логи контейнера
2. Убедитесь в корректности API ключей
3. Проверьте доступность внешних сервисов
4. Обратитесь к документации Azure и Vercel 