# Scientific API для анализа крупномасштабной структуры Вселенной

API для работы с каталогами галактик и астрономическими данными, построения графов и анализа топологических свойств крупномасштабной структуры Вселенной.

## Возможности API

- Работа с каталогами галактик (SDSS, Euclid, DESI, DES)
- Поиск научных статей через API ADS (NASA Astrophysics Data System)
- Управление датасетами и файлами
- Применение методов машинного обучения к астрономическим данным
- Анализ топологических свойств графов крупномасштабной структуры Вселенной

## Запуск локально

1. Клонируйте репозиторий:
```
git clone https://github.com/yourusername/scientific-api.git
cd scientific-api
```

2. Создайте и активируйте виртуальное окружение:
```
python -m venv .venv
source .venv/bin/activate  # На Linux/Mac
# или
.venv\Scripts\activate  # На Windows
```

3. Установите зависимости:
```
pip install -r requirements.txt
```

4. Запустите сервер:
```
uvicorn api.index:app --reload
```

5. Откройте в браузере: http://localhost:8000

## Деплой на Vercel

1. Зарегистрируйтесь на [Vercel](https://vercel.com)

2. Установите Vercel CLI:
```
npm i -g vercel
```

3. Войдите в свой аккаунт:
```
vercel login
```

4. Выполните деплой:
```
vercel
```

5. Для деплоя в продакшн:
```
vercel --prod
```

## Структура проекта

- `api/` - FastAPI роутеры и серверный код
- `ui/` - HTML, CSS, JavaScript для веб-интерфейса
- `utils/` - Вспомогательные функции и утилиты
- `galaxy_data/` - Каталоги и данные (не включены в репозиторий из-за размера)

## Документация API

После запуска API, документация Swagger доступна по адресу:
- http://localhost:8000/docs (локально)
- https://your-vercel-domain.vercel.app/docs (на Vercel)

## Требования

- Python 3.9+
- FastAPI
- Pandas
- NumPy
- scikit-learn
- Другие зависимости перечислены в requirements.txt

## Решение проблем с деплоем на Vercel

1. Для успешного деплоя на Vercel убедитесь, что:
   - Все зависимости указаны в requirements-vercel.txt
   - В vercel.json настроены правильные пути
   - Файл api/index.py является точкой входа
   - Файл .vercelignore содержит правильные исключения

2. При ошибках после деплоя:
   - Проверьте логи в панели Vercel
   - Используйте `vercel logs` для получения подробной информации
   - Убедитесь, что размер лямбда-функции не превышает 50 МБ (ограничение Vercel)

This project is split into two parts:
1. Lightweight API (deployed to Vercel)
2. Heavy Compute Service (deployed to a separate platform)

## Lightweight API (Vercel)

The lightweight API provides basic endpoints and serves the frontend. It's deployed to Vercel and has minimal dependencies.

### Dependencies
- FastAPI
- Uvicorn
- Python-dotenv
- Jinja2
- Aiofiles
- Requests
- Pydantic

### Deployment
1. Make sure you have the Vercel CLI installed:
```bash
npm install -g vercel
```

2. Deploy to Vercel:
```bash
vercel --prod
```

## Heavy Compute Service

The heavy compute service handles data processing, machine learning, and other resource-intensive operations. It should be deployed to a platform without strict size limitations (e.g., Google Cloud Run, AWS Lambda Container, or Heroku).

### Dependencies
See `requirements-heavy.txt` for the full list of dependencies, including:
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Google API Client
- And more...

### Deployment Options

#### Google Cloud Run
1. Build the Docker image:
```bash
docker build -t scientific-api-heavy .
```

2. Push to Google Container Registry:
```bash
docker tag scientific-api-heavy gcr.io/[PROJECT_ID]/scientific-api-heavy
docker push gcr.io/[PROJECT_ID]/scientific-api-heavy
```

3. Deploy to Cloud Run:
```bash
gcloud run deploy scientific-api-heavy \
  --image gcr.io/[PROJECT_ID]/scientific-api-heavy \
  --platform managed \
  --allow-unauthenticated
```

#### AWS Lambda Container
1. Build the Docker image:
```bash
docker build -t scientific-api-heavy .
```

2. Create an ECR repository:
```bash
aws ecr create-repository --repository-name scientific-api-heavy
```

3. Push to ECR:
```bash
aws ecr get-login-password | docker login --username AWS --password-stdin [ACCOUNT_ID].dkr.ecr.[REGION].amazonaws.com
docker tag scientific-api-heavy [ACCOUNT_ID].dkr.ecr.[REGION].amazonaws.com/scientific-api-heavy:latest
docker push [ACCOUNT_ID].dkr.ecr.[REGION].amazonaws.com/scientific-api-heavy:latest
```

4. Create a Lambda function using the container image.

## Environment Variables

### Vercel (Lightweight API)
- `ENVIRONMENT`: "production" or "development"
- `HEAVY_COMPUTE_URL`: URL of the heavy compute service

### Heavy Compute Service
- `ADSABS_TOKEN`: Token for ADS API
- `GOOGLE_CLIENT_ID`: Google API client ID
- `GOOGLE_CLIENT_SECRET`: Google API client secret
- `GOOGLE_REFRESH_TOKEN`: Google API refresh token
- `SERPAPI_KEY`: Key for SerpAPI
- `EUCLID_URL`: URL for Euclid data

## Development

1. Install dependencies for the lightweight API:
```bash
pip install -r requirements.txt
```

2. Install dependencies for the heavy compute service:
```bash
pip install -r requirements-heavy.txt
```

3. Run the lightweight API locally:
```bash
uvicorn api.index:app --reload
```

4. Run the heavy compute service locally:
```bash
uvicorn api.heavy_api:app --reload --port 8001
```

## Testing

1. Test the lightweight API:
```bash
pytest tests/test_light_api.py
```

2. Test the heavy compute service:
```bash
pytest tests/test_heavy_api.py 