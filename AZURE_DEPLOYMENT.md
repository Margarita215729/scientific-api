# Развертывание Scientific API в Azure Container Instances

## Обзор

Этот проект содержит скрипты для развертывания Scientific API в Azure Container Instances с поддержкой различных конфигураций.

## Доступные скрипты развертывания

### 1. Проверка окружения
```bash
./deploy/check-docker-images.sh
```
Проверяет наличие Docker, Azure CLI и локальных образов перед развертыванием.

### 2. Упрощенная версия API
```bash
./deploy/azure-amd64.sh
```
Развертывает базовую версию API с минимальным набором функций:
- `/ping` - проверка работоспособности
- `/ads/*` - поиск по литературе ADS
- `/astro/*` - простые астрономические данные

### 3. Полная версия API
```bash
./deploy/azure-amd64-full.sh
```
Развертывает полную версию API со всеми функциями:
- Все функции упрощенной версии
- `/astro/full/*` - полные астрономические каталоги (SDSS, Gaia, DESI, etc.)
- `/datasets/*` - управление наборами данных
- `/files/*` - файловые операции
- `/ml/*` - модели машинного обучения
- `/analysis/*` - анализ данных

### 4. Развертывание с локальным образом ARM64
```bash
./deploy/azure-local-image.sh
```
Использует существующий локальный образ (может не работать из-за несовместимости архитектур).

## Доступные эндпоинты

### Основные
- `GET /ping` - проверка работоспособности
- `GET /api` - информация об API
- `GET /docs` - интерактивная документация

### Астрономические данные
- `GET /astro/statistics` - статистика каталогов
- `GET /astro/status` - статус каталогов
- `GET /astro/galaxies` - информация о галактиках

### Литература ADS
- `GET /ads/search-by-object?object_name={name}` - поиск по объекту
- `GET /ads/search-by-coordinates?ra={ra}&dec={dec}&radius={radius}` - поиск по координатам
- `GET /ads/citations?bibcode={bibcode}` - получение цитирований

### Наборы данных
- `GET /datasets/academic` - академические наборы данных
- `GET /datasets/nasa` - наборы данных NASA
- `GET /datasets/arxiv` - данные ArXiv

### Файловые операции
- `POST /files/upload` - загрузка файлов
- `GET /files/list` - список файлов
- `GET /files/read` - чтение файлов

### Машинное обучение
- `POST /ml/train` - обучение моделей

## Управление развертыванием

### Просмотр логов
```bash
az container logs --resource-group scientific-api --name scientific-api-full --follow
```

### Проверка статуса
```bash
az container show --resource-group scientific-api --name scientific-api-full --query 'instanceView.state'
```

### Удаление контейнера
```bash
az container delete --resource-group scientific-api --name scientific-api-full --yes
```

### Удаление образа из ACR
```bash
az acr repository delete --name scientificapiacr --image scientific-api-full:latest --yes
```

## Решение проблем

### Проблема с архитектурой
Если получаете ошибку `ImageOsTypeNotMatchContainerGroup`, используйте скрипт `azure-amd64-full.sh` вместо `azure-local-image.sh`.

### Недоступные каталоги
Если каталоги показывают статус "Недоступен", проверьте логи на наличие ошибок импорта:
```bash
az container logs --resource-group scientific-api --name scientific-api-full | grep "Warning"
```

### Превышение квоты ресурсов
API настроен на использование минимальных ресурсов (1 CPU, 1.5GB RAM) для совместимости с бесплатным уровнем Azure.

## Зависимости

Полная версия API требует следующих Python пакетов:
- fastapi, uvicorn
- pandas, numpy, scikit-learn
- matplotlib, seaborn
- astropy, astroquery
- google-api-python-client
- requests, pydantic

## Архитектура

API построен с использованием:
- **FastAPI** - веб-фреймворк
- **Docker** - контейнеризация
- **Azure Container Registry** - хранение образов
- **Azure Container Instances** - развертывание
- **Модульная архитектура** - отдельные роутеры для разных функций 