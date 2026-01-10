"""Central configuration management using pydantic Settings."""

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field, MongoDsn
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # MongoDB Configuration
    MONGO_URI: MongoDsn = Field(
        default="mongodb://localhost:27017",
        description="MongoDB connection URI",
    )
    MONGO_DB_NAME: str = Field(
        default="scientific_api",
        description="MongoDB database name",
    )

    # Redis Configuration
    REDIS_URL: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL",
    )

    # Logging Configuration
    LOG_LEVEL: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )

    # Data Directories
    DATA_ROOT: Path = Field(
        default=Path("/workspaces/scientific-api/data"),
        description="Root directory for data storage",
    )

    # Application Configuration
    APP_NAME: str = Field(
        default="Scientific API",
        description="Application name",
    )
    APP_VERSION: str = Field(
        default="0.1.0",
        description="Application version",
    )
    DEBUG: bool = Field(
        default=False,
        description="Debug mode flag",
    )

    # API Configuration
    API_V1_PREFIX: str = Field(
        default="/api/v1",
        description="API v1 route prefix",
    )
    CORS_ORIGINS: list[str] = Field(
        default=["*"],
        description="Allowed CORS origins",
    )

    # ML Configuration
    ML_RANDOM_SEED: int = Field(
        default=42,
        description="Random seed for reproducibility",
    )
    ML_N_JOBS: int = Field(
        default=-1,
        description="Number of parallel jobs for ML tasks (-1 for all cores)",
    )

    # Celery Configuration (for async tasks)
    CELERY_BROKER_URL: Optional[str] = Field(
        default=None,
        description="Celery broker URL (uses REDIS_URL if not set)",
    )
    CELERY_RESULT_BACKEND: Optional[str] = Field(
        default=None,
        description="Celery result backend URL (uses REDIS_URL if not set)",
    )

    def get_celery_broker_url(self) -> str:
        """Get Celery broker URL, fallback to Redis URL."""
        return self.CELERY_BROKER_URL or self.REDIS_URL

    def get_celery_result_backend(self) -> str:
        """Get Celery result backend URL, fallback to Redis URL."""
        return self.CELERY_RESULT_BACKEND or self.REDIS_URL


@lru_cache
def get_settings() -> Settings:
    """
    Get cached application settings.

    Returns:
        Settings: Application configuration instance.
    """
    return Settings()
