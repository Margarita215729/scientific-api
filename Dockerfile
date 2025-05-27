# Multi-stage build for Scientific API
# Supports both lightweight (Vercel proxy) and full (Azure backend) deployments
FROM python:3.11-slim as builder

# Build arguments - Azure is default for production
ARG BUILD_TYPE=azure
# MAIN_FILE теперь будет определяться командой запуска в Azure, а для Vercel - vercel.json
# ARG MAIN_FILE=main_azure_with_db.py 

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    pkg-config \
    curl \
    wget \
    git \
    sqlite3 \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements files
# Всегда устанавливаем полные зависимости, так как образ теперь один для Azure
COPY requirements_azure.txt requirements.txt 
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim

# Install runtime dependencies (меньший набор, т.к. компиляторы не нужны)
RUN apt-get update && apt-get install -y \
    libopenblas0 \
    liblapack3 \
    curl \
    sqlite3 \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Create app user 
RUN useradd --create-home --shell /bin/bash app

WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY . .

# Create directories 
RUN mkdir -p /app/database galaxy_data/processed galaxy_data/cache galaxy_data/ml_ready && \
    chown -R app:app /app galaxy_data

# Set environment variables
ENV PYTHONPATH=/app
ENV ENVIRONMENT=production
ENV PYTHONUNBUFFERED=1
# PORT будет установлен платформой Azure Web App или локально через docker run
# ENV PORT=8000 

# Switch to app user
USER app

# Expose port (для информации Docker, Azure Web App сам управляет маппингом)
EXPOSE 8000

# Health check - теперь более общий, так как CMD изменится
HEALTHCHECK --interval=30s --timeout=30s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/ping || exit 1

# Default CMD (для Azure, будет запускать main_azure_with_db.py)
# Команда запуска для Azure будет задана через --startup-file в deploy_azure_bicep.sh
# Оставляем простой CMD, который может быть переопределен.
# Если Azure не переопределит, этот CMD запустит uvicorn с main.py, что мы не хотим для Azure.
# Поэтому важно, чтобы --startup-file в Azure был настроен.
# Для локального тестирования Azure-версии: docker run ... image_name python main_azure_with_db.py
CMD echo "Default CMD: To run the Azure backend, specify 'python main_azure_with_db.py' as startup command." && uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 