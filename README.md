# Scientific API - Astronomical Data Processing Platform

## 🌟 Обзор проекта

Scientific API - это современная платформа для обработки и анализа астрономических данных, развернутая в Azure Cloud с использованием Docker контейнеров и Bicep templates.

### ✨ Основные возможности

- 🔭 **Астрономические данные**: Доступ к каталогам SDSS, DESI, DES, Euclid
- 🤖 **Машинное обучение**: ML-готовые датасеты и модели
- 📊 **Анализ данных**: Статистический анализ и визуализация
- 🔍 **Поиск публикаций**: Интеграция с NASA ADS
- 🌐 **Web интерфейс**: Современный UI для работы с данными
- 🔒 **Безопасность**: HTTPS, VNet интеграция, отключенные FTP/SCM

## 🏗️ Архитектура

### Azure Resources
- **Web App**: `scientific-api` (Canada Central)
- **Docker Image**: `index.docker.io/gretk/scientific-api-app-image:scientific-api`
- **Cosmos DB**: `scientific-api-server` / `scientific-data`
- **VNet**: `vnet-euoxdfir` / `subnet-nwivqmzl`
- **URL**: https://scientific-api-e3a7a5dph6b3axa3.canadacentral-01.azurewebsites.net

### Технологический стек
- **Backend**: FastAPI (Python)
- **Frontend**: HTML/CSS/JavaScript
- **Database**: Azure Cosmos DB (MongoDB API)
- **Container**: Docker
- **Infrastructure**: Azure Bicep/ARM Templates
- **CI/CD**: Azure CLI scripts

## 🚀 Быстрый старт

### Предварительные требования

```bash
# Azure CLI
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

# Docker
sudo apt-get update
sudo apt-get install docker.io

# Python 3.8+
python3 --version
```

### Развертывание

#### Вариант 1: Bicep Template (рекомендуется)
```bash
# Клонируем репозиторий
git clone <repository-url>
cd scientific-api

# Логинимся в Azure
az login

# Развертываем с помощью Bicep
chmod +x deploy_azure_bicep.sh
./deploy_azure_bicep.sh
```

#### Вариант 2: ARM Template
```bash
chmod +x deploy_azure_webapp.sh
./deploy_azure_webapp.sh
```

#### Вариант 3: Container Instances
```bash
chmod +x deploy_azure.sh
./deploy_azure.sh
```

### Локальная разработка

```bash
# Создаем виртуальное окружение
python3 -m venv .venv
source .venv/bin/activate

# Устанавливаем зависимости
pip install -r requirements.txt

# Запускаем локально
python main.py
```

### Docker

```bash
# Сборка образа
docker build -t scientific-api .

# Запуск контейнера
docker run -p 8000:8000 scientific-api

# Или используем docker-compose
docker-compose up
```

## 📁 Структура проекта

```
scientific-api/
├── api/                          # API модули
│   ├── heavy_api.py             # Тяжелые вычисления
│   ├── cosmos_db_config.py      # Конфигурация Cosmos DB
│   └── logging_setup.py         # Настройка логирования
├── utils/                        # Утилиты
│   ├── ads_astronomy_real.py    # NASA ADS интеграция
│   └── astronomy_catalogs.py    # Астрономические каталоги
├── ui/                          # Web интерфейс
│   ├── index.html              # Главная страница
│   └── ads.html                # ADS поиск
├── database/                    # База данных
│   └── schema.sql              # SQL схема
├── azure-webapp-bicep.bicep    # Bicep template
├── azure-webapp-config.json    # ARM template
├── deploy_azure_bicep.sh       # Скрипт развертывания (Bicep)
├── deploy_azure_webapp.sh      # Скрипт развертывания (ARM)
├── docker-compose.yml          # Docker Compose
├── Dockerfile                  # Docker образ
├── main.py                     # Главный файл приложения
├── azure.env                   # Переменные окружения
└── requirements.txt            # Python зависимости
```

## 🔧 Конфигурация

### Переменные окружения

Основные переменные находятся в `azure.env`:

```bash
# Azure Configuration
AZURE_SUBSCRIPTION_ID=8e746503-c0c0-4535-a05d-49e544196e3f
AZURE_RESOURCE_GROUP=scientific-api
AZURE_APP_NAME=scientific-api
AZURE_APP_URL=https://scientific-api-e3a7a5dph6b3axa3.canadacentral-01.azurewebsites.net

# Cosmos DB
COSMOS_DB_ACCOUNT=scientific-api-server
COSMOS_DB_DATABASE=scientific-data

# Network
VNET_NAME=vnet-euoxdfir
SUBNET_NAME=subnet-nwivqmzl
```

### API ключи (устанавливаются отдельно)

```bash
az webapp config appsettings set \
    --resource-group scientific-api \
    --name scientific-api \
    --settings \
        GOOGLE_CLIENT_ID="your_client_id" \
        GOOGLE_CLIENT_SECRET="your_client_secret" \
        ADSABS_TOKEN="your_adsabs_token" \
        SERPAPI_KEY="your_serpapi_key" \
        COSMOS_DB_KEY="your_cosmos_db_key"
```

## 📚 API Документация

### Основные эндпоинты

| Эндпоинт | Описание |
|----------|----------|
| `/ping` | Health check |
| `/docs` | Swagger документация |
| `/api/astro/galaxies` | Данные галактик |
| `/api/astro/statistics` | Статистика |
| `/api/ads/search` | Поиск публикаций |
| `/api/ml/prepare-dataset` | Подготовка ML датасета |

### Примеры использования

```bash
# Health check
curl https://scientific-api-e3a7a5dph6b3axa3.canadacentral-01.azurewebsites.net/ping

# Получить данные галактик
curl "https://scientific-api-e3a7a5dph6b3axa3.canadacentral-01.azurewebsites.net/api/astro/galaxies?limit=10&min_z=0.1"

# Поиск в ADS
curl "https://scientific-api-e3a7a5dph6b3axa3.canadacentral-01.azurewebsites.net/api/ads/search?query=galaxy+formation"
```

## 🔒 Безопасность

### Настройки безопасности
- ✅ HTTPS Only (принудительно)
- ✅ TLS 1.2+ (минимальная версия)
- ✅ FTP доступ отключен
- ✅ SCM доступ отключен
- ✅ VNet интеграция включена
- ✅ Managed Identity для Key Vault

### Сетевая безопасность
- VNet: `vnet-euoxdfir`
- Subnet: `subnet-nwivqmzl`
- Private DNS: `privatelink.mongo.cosmos.azure.com`

## 📊 Мониторинг и управление

### Команды управления

```bash
# Просмотр логов
az webapp log tail --resource-group scientific-api --name scientific-api

# Перезапуск приложения
az webapp restart --resource-group scientific-api --name scientific-api

# Просмотр конфигурации
az webapp config show --resource-group scientific-api --name scientific-api

# Масштабирование
az webapp config set --resource-group scientific-api --name scientific-api --number-of-workers 2
```

### Метрики и алерты

Доступны через Azure Portal:
- CPU utilization
- Memory usage
- Request count
- Response time
- Error rate

## 🧪 Тестирование

### Локальное тестирование

```bash
# Запуск тестов
python -m pytest tests/

# Проверка health endpoint
curl http://localhost:8000/ping

# Проверка API
curl http://localhost:8000/api/astro/status
```

### Production тестирование

```bash
# Health check
curl -f https://scientific-api-e3a7a5dph6b3axa3.canadacentral-01.azurewebsites.net/ping

# Load testing
ab -n 100 -c 10 https://scientific-api-e3a7a5dph6b3axa3.canadacentral-01.azurewebsites.net/ping
```

## 🔄 CI/CD

### Автоматическое развертывание

1. **GitHub Actions** (рекомендуется)
2. **Azure DevOps Pipelines**
3. **Manual deployment** с помощью скриптов

### Workflow

```yaml
# .github/workflows/deploy.yml
name: Deploy to Azure
on:
  push:
    branches: [main]
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Deploy to Azure
        run: ./deploy_azure_bicep.sh
```

## 📈 Производительность

### Оптимизации
- Docker multi-stage builds
- Кэширование в Cosmos DB
- Асинхронные операции
- Connection pooling
- Gzip compression

### Масштабирование
- Horizontal scaling (multiple instances)
- Auto-scaling rules
- Load balancing
- CDN для статических файлов

## 🐛 Устранение неполадок

### Частые проблемы

1. **Container не запускается**
   ```bash
   az webapp log tail --resource-group scientific-api --name scientific-api
   ```

2. **Ошибки подключения к Cosmos DB**
   ```bash
   az webapp config appsettings list --resource-group scientific-api --name scientific-api
   ```

3. **Проблемы с VNet**
   ```bash
   az network vnet subnet show --resource-group scientific-api --vnet-name vnet-euoxdfir --name subnet-nwivqmzl
   ```

### Логи и диагностика

```bash
# Application logs
az webapp log tail --resource-group scientific-api --name scientific-api

# Deployment logs
az webapp deployment log list --resource-group scientific-api --name scientific-api

# Container logs
az webapp log download --resource-group scientific-api --name scientific-api
```

## 🤝 Участие в разработке

### Требования для разработки
- Python 3.8+
- Docker
- Azure CLI
- Git

### Процесс разработки
1. Fork репозитория
2. Создайте feature branch
3. Внесите изменения
4. Добавьте тесты
5. Создайте Pull Request

## 📄 Лицензия

MIT License - см. файл LICENSE

## 📞 Поддержка

- **Issues**: GitHub Issues
- **Documentation**: `/docs` endpoint
- **API Reference**: Swagger UI

---

**Статус проекта**: ✅ Production Ready

**Последнее обновление**: $(date)

**Версия**: 2.0.0 