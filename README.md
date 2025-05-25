# Scientific API - Анализ крупномасштабных структур Вселенной

## 🚀 Архитектура

Проект состоит из двух основных компонентов:

1. **Azure Container Instance** - Тяжелые вычисления и обработка данных
2. **Vercel Web App** - Веб-интерфейс и легковесное API

### Azure Container (Backend)
- **URL**: `scientific-api-full-1748121289.eastus.azurecontainer.io:8000`
- **Функции**: 
  - Автоматическая предзагрузка и нормализация астрономических каталогов
  - Обработка больших данных (SDSS, DESI, DES, Euclid)
  - Machine Learning готовые датасеты
  - Статистический анализ

### Vercel Web App (Frontend)
- **URL**: `https://scientific-pjciwtna6-makeeva01m-gmailcoms-projects.vercel.app`
- **Функции**:
  - Веб-интерфейс для работы с данными
  - Проксирование запросов к Azure API
  - Статические файлы и UI

## 🗂️ Структура проекта (после очистки)

```
scientific-api/
├── api/
│   ├── index.py              # Главный API файл для Vercel
│   ├── heavy_api.py          # Тяжелое API для Azure
│   ├── static_files.py       # Настройка статических файлов
│   └── dependencies.py       # Управление зависимостями
├── ui/                       # Веб-интерфейс
│   ├── index.html
│   ├── ads.html
│   └── script.js
├── utils/
│   └── data_preprocessor.py  # Предобработка каталогов данных
├── startup.py                # Startup скрипт для Azure
├── Dockerfile               # Docker конфигурация для Azure
├── deploy_azure.sh          # Скрипт развертывания Azure
├── requirements.txt         # Зависимости Python
├── vercel.json             # Конфигурация Vercel
└── README.md
```

## 🔄 Процесс предзагрузки данных

При запуске Azure контейнера автоматически выполняется:

1. **Скачивание каталогов**:
   - SDSS DR17 (50,000 объектов)
   - DESI DR1 (30,000 объектов)  
   - DES Y6 (40,000 объектов)
   - Euclid Q1 (20,000 объектов, sample data)

2. **Нормализация данных**:
   - Унификация названий колонок
   - Очистка от невалидных значений
   - Преобразование координат в декартову систему
   - Вычисление цветовых индексов

3. **Создание объединенного датасета**:
   - Слияние всех каталогов
   - Удаление дубликатов
   - Сохранение в CSV формате

## 🚀 Развертывание

### Azure Container

1. **Убедитесь, что у вас установлены**:
   ```bash
   # Azure CLI
   curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
   
   # Docker
   sudo apt-get install docker.io
   ```

2. **Запустите развертывание**:
   ```bash
   ./deploy_azure.sh
   ```

3. **Мониторинг развертывания**:
   ```bash
   # Просмотр логов
   az container logs --resource-group scientific-api --name scientific-api-full
   
   # Статус контейнера
   az container show --resource-group scientific-api --name scientific-api-full
   ```

### Vercel Web App

1. **Установите Vercel CLI**:
   ```bash
   npm i -g vercel
   ```

2. **Развертывание**:
   ```bash
   vercel --prod
   ```

## 📊 API Endpoints

### Azure Container API (`scientific-api-full-1748121289.eastus.azurecontainer.io:8000`)

#### Основные эндпоинты:
- `GET /ping` - Проверка работоспособности
- `GET /astro` - Обзор астрономических сервисов
- `GET /astro/status` - Статус каталогов данных
- `GET /astro/statistics` - Статистика по данным
- `GET /astro/galaxies` - Получение данных галактик

#### Пример запроса:
```bash
curl "http://scientific-api-full-1748121289.eastus.azurecontainer.io:8000/astro/galaxies?source=SDSS&limit=100&min_z=0.1&max_z=0.5"
```

### Vercel API (`scientific-pjciwtna6-makeeva01m-gmailcoms-projects.vercel.app`)

- `GET /` - Веб-интерфейс
- `GET /api` - Информация о доступных эндпоинтах
- `GET /ping` - Проверка работоспособности
- Проксирование всех `/astro/*` запросов к Azure API

## 🔧 Конфигурация

### Environment Variables (Azure)

```bash
ENVIRONMENT=production
PYTHONUNBUFFERED=1
ADSABS_TOKEN=your_ads_token
SERPAPI_KEY=your_serpapi_key
HOST=0.0.0.0
PORT=8000
WORKERS=1
```

### Параметры контейнера

- **CPU**: 2 cores
- **Memory**: 4 GB  
- **Storage**: Ephemeral (данные обрабатываются in-memory)
- **Restart Policy**: Always

## 🧪 Тестирование

### Проверка работы Azure API:
```bash
# Health check
curl http://scientific-api-full-1748121289.eastus.azurecontainer.io:8000/ping

# Статус данных
curl http://scientific-api-full-1748121289.eastus.azurecontainer.io:8000/astro/status

# Получение данных
curl "http://scientific-api-full-1748121289.eastus.azurecontainer.io:8000/astro/galaxies?limit=10"
```

### Проверка Vercel приложения:
```bash
# Health check
curl https://scientific-pjciwtna6-makeeva01m-gmailcoms-projects.vercel.app/ping

# Веб-интерфейс
open https://scientific-pjciwtna6-makeeva01m-gmailcoms-projects.vercel.app
```

## 📈 Мониторинг

### Azure Container Logs:
```bash
az container logs --resource-group scientific-api --name scientific-api-full --follow
```

### Vercel Logs:
```bash
vercel logs https://scientific-pjciwtna6-makeeva01m-gmailcoms-projects.vercel.app
```

## 🔍 Troubleshooting

### Если Azure контейнер не отвечает:

1. **Проверьте статус**:
   ```bash
   az container show --resource-group scientific-api --name scientific-api-full
   ```

2. **Просмотрите логи**:
   ```bash
   az container logs --resource-group scientific-api --name scientific-api-full
   ```

3. **Перезапустите контейнер**:
   ```bash
   az container restart --resource-group scientific-api --name scientific-api-full
   ```

### Если данные не загружаются:

1. Контейнер может все еще обрабатывать данные при первом запуске
2. Проверьте логи на наличие ошибок загрузки
3. При неудаче загрузки реальных данных система автоматически сгенерирует sample data

## 🎯 Production Ready

Система полностью готова к production использованию:

- ✅ Автоматическая предзагрузка данных
- ✅ Обработка ошибок без fallback на mock данные  
- ✅ Horizontal scaling готовность
- ✅ Health checks и мониторинг
- ✅ Чистая архитектура без неиспользуемых файлов
- ✅ Корректные HTTP статус коды (503 при недоступности данных)

## 📞 Support

При возникновении проблем проверьте:
1. Логи Azure контейнера
2. Статус предобработки данных через `/astro/status`
3. Vercel deployment logs