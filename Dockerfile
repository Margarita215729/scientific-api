# Multi-stage build for Scientific API
# Supports both lightweight (Vercel proxy) and full (Azure backend) deployments
FROM python:3.11-slim as builder

# Build arguments - Azure is default for production
ARG BUILD_TYPE=azure
ARG MAIN_FILE=main_azure_with_db.py

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
    && if [ "$BUILD_TYPE" = "azure" ]; then \
        apt-get install -y sqlite3 postgresql-client; \
    fi \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements files
COPY requirements.txt .
RUN if [ "$BUILD_TYPE" = "azure" ]; then \
        if [ -f requirements_azure.txt ]; then \
            cp requirements_azure.txt requirements_full.txt; \
        else \
            echo "fastapi==0.104.1" > requirements_full.txt && \
            echo "uvicorn==0.24.0" >> requirements_full.txt && \
            echo "httpx==0.25.2" >> requirements_full.txt && \
            echo "pandas==2.2.2" >> requirements_full.txt && \
            echo "numpy==1.26.4" >> requirements_full.txt && \
            echo "scipy==1.13.0" >> requirements_full.txt && \
            echo "scikit-learn==1.4.2" >> requirements_full.txt && \
            echo "matplotlib==3.8.4" >> requirements_full.txt && \
            echo "astroquery==0.4.7" >> requirements_full.txt && \
            echo "azure-cosmos==4.5.1" >> requirements_full.txt && \
            echo "asyncpg==0.29.0" >> requirements_full.txt && \
            echo "python-dotenv==1.0.1" >> requirements_full.txt; \
        fi && \
        pip install --no-cache-dir --upgrade pip && \
        pip install --no-cache-dir -r requirements_full.txt; \
    else \
        pip install --no-cache-dir --upgrade pip && \
        pip install --no-cache-dir -r requirements.txt; \
    fi

# Production stage
FROM python:3.11-slim

# Build arguments (need to redeclare in new stage)
ARG BUILD_TYPE=azure
ARG MAIN_FILE=main_azure_with_db.py

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libopenblas0 \
    liblapack3 \
    curl \
    && if [ "$BUILD_TYPE" = "azure" ]; then \
        apt-get install -y sqlite3 postgresql-client; \
    fi \
    && rm -rf /var/lib/apt/lists/*

# Create app user for Azure builds
RUN if [ "$BUILD_TYPE" = "azure" ]; then \
        useradd --create-home --shell /bin/bash app; \
    fi

# Set working directory
WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY . .

# Create directories based on build type
RUN if [ "$BUILD_TYPE" = "azure" ]; then \
        mkdir -p /app/database galaxy_data/processed galaxy_data/cache galaxy_data/ml_ready && \
        chown -R app:app /app; \
    else \
        mkdir -p galaxy_data/processed galaxy_data/cache galaxy_data/ml_ready; \
    fi

# Create database initialization script for Azure builds
RUN if [ "$BUILD_TYPE" = "azure" ]; then \
        echo '#!/usr/bin/env python3\n\
import asyncio\n\
import sys\n\
sys.path.append("/app")\n\
from database.config import db\n\
\n\
async def init_db():\n\
    try:\n\
        await db.init_database()\n\
        print("Database initialized successfully")\n\
    except Exception as e:\n\
        print(f"Database initialization failed: {e}")\n\
\n\
if __name__ == "__main__":\n\
    asyncio.run(init_db())\n\
' > /app/init_db.py && chmod +x /app/init_db.py && chown app:app /app/init_db.py; \
    fi

# Set environment variables
ENV PYTHONPATH=/app
ENV ENVIRONMENT=production
ENV PYTHONUNBUFFERED=1

# Set Azure-specific environment variables
RUN if [ "$BUILD_TYPE" = "azure" ]; then \
        echo "ENV HEAVY_PIPELINE_ON_START=true" >> /etc/environment; \
    fi

# Switch to app user for Azure builds
RUN if [ "$BUILD_TYPE" = "azure" ]; then \
        echo "Switching to app user for Azure build"; \
    fi
USER ${BUILD_TYPE:+app}

# Expose port
EXPOSE 8000

# Health check with fallback
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/ping || curl -f http://localhost:8000/api/health || exit 1

# Dynamic start command based on build type
CMD if [ "$BUILD_TYPE" = "azure" ]; then \
        echo "Starting Azure backend with database..." && \
        python main_azure_with_db.py; \
    else \
        echo "Starting lightweight proxy..." && \
        uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4; \
    fi 