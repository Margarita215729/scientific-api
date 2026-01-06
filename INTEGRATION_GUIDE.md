# Integration Guide for Experiment API

## Overview

This guide explains how to integrate the experiment management API into the main FastAPI application.

## 1. Register Routes in Main App

Add the following to your main FastAPI application file (e.g., `api/index.py` or `main.py`):

```python
from fastapi import FastAPI
from app.api.routes import experiments
from app.core.config import get_settings

settings = get_settings()

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
)

# Register experiment routes
app.include_router(
    experiments.router,
    prefix=settings.API_V1_PREFIX,
    tags=["experiments"],
)
```

## 2. Initialize MongoDB Indexes

Add to application startup event:

```python
from app.db.experiments import create_indexes

@app.on_event("startup")
async def startup_event():
    """Initialize database indexes on startup."""
    await create_indexes()
    logger.info("MongoDB indexes created")
```

## 3. Start Celery Worker

In a separate terminal or process, start the Celery worker:

```bash
# Development mode
celery -A app.services.tasks worker --loglevel=info

# Production mode (with concurrency control)
celery -A app.services.tasks worker --loglevel=info --concurrency=4 --pool=prefork
```

## 4. (Optional) Start Celery Beat for Periodic Tasks

If you want periodic cleanup tasks:

```bash
celery -A app.services.tasks beat --loglevel=info
```

**Note**: You need to configure beat schedule in `app/services/tasks.py` first.

## 5. Environment Variables

Ensure the following environment variables are set in `.env`:

```env
# MongoDB
MONGO_URI=mongodb://localhost:27017
MONGO_DB_NAME=scientific_api

# Redis (for Celery)
REDIS_URL=redis://localhost:6379/0

# Optional: Celery-specific URLs (defaults to REDIS_URL)
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0

# Data directories
DATA_ROOT=/workspaces/scientific-api/data
```

## 6. Test the API

### Create an Experiment

```bash
curl -X POST "http://localhost:8000/api/v1/experiments/" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Test Experiment",
    "description": "Testing the experiment API",
    "config": {
      "cosmology_config_path": "/configs/cosmology_sdss_dr18.yaml",
      "quantum_config_path": "/configs/quantum_heisenberg_2d.yaml",
      "n_cosmology_graphs": 5,
      "n_quantum_graphs": 5
    },
    "tags": ["test"]
  }'
```

### Run Experiment (Async)

```bash
curl -X POST "http://localhost:8000/api/v1/experiments/{experiment_id}/run-async"
```

### Check Status

```bash
curl -X GET "http://localhost:8000/api/v1/experiments/{experiment_id}/status"
```

### Get Results

```bash
curl -X GET "http://localhost:8000/api/v1/experiments/{experiment_id}/results"
```

### List Experiments

```bash
curl -X GET "http://localhost:8000/api/v1/experiments/?limit=10&skip=0"
```

## 7. OpenAPI Documentation

Once routes are registered, FastAPI will automatically generate interactive documentation at:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 8. Production Considerations

### Database

- Use MongoDB replica set for high availability
- Enable authentication and SSL/TLS
- Create proper indexes (done automatically on startup)

### Celery

- Use proper message broker (RabbitMQ or Redis with persistence)
- Configure retry policies and timeouts
- Monitor task queues and worker health
- Use flower for Celery monitoring:
  ```bash
  celery -A app.services.tasks flower
  ```

### API

- Add authentication/authorization middleware
- Implement rate limiting
- Add request validation and sanitization
- Use reverse proxy (nginx/traefik) for SSL termination
- Enable CORS with specific origins (not "*")

### Monitoring

- Add Prometheus metrics
- Log all API requests and responses
- Monitor experiment execution times
- Track Celery task metrics

## 9. File Structure After Integration

```
/workspaces/scientific-api/
├── app/
│   ├── api/
│   │   ├── routes/
│   │   │   └── experiments.py       # ✅ Created
│   │   └── schemas/
│   │       └── experiments.py        # ✅ Created
│   ├── core/
│   │   ├── config.py                 # ✅ Existing (used by all)
│   │   └── logging.py
│   ├── db/
│   │   └── experiments.py            # ✅ Created
│   └── services/
│       ├── experiment_runner.py      # ✅ Created
│       └── tasks.py                  # ✅ Created
├── api/
│   └── index.py                      # ⚠️ UPDATE: Register routes here
├── main.py                           # ⚠️ Or here (if this is entry point)
├── data/
│   └── experiments/                  # Auto-created by runner
│       └── {experiment_id}/
│           ├── graphs/
│           ├── features/
│           ├── models/
│           └── distances/
└── requirements.txt                  # ✅ Updated
```

## 10. Next Steps

- [ ] Register routes in main app
- [ ] Add authentication/authorization
- [ ] Create startup script for MongoDB indexes
- [ ] Add visualization endpoints
- [ ] Implement unit tests
- [ ] Create integration tests
- [ ] Add API documentation examples
- [ ] Deploy with Docker Compose (app + MongoDB + Redis + Celery worker)
