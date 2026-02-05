"""Application configuration using Pydantic Settings."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, PostgresDsn, RedisDsn
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ============== Environment ==============
    environment: Literal["development", "staging", "production"] = "development"

    # ============== Database ==============
    postgres_user: str = "lifeai"
    postgres_password: str = "lifeai_dev_password"
    postgres_db: str = "lifeai_kg"
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    database_url: PostgresDsn | None = None

    @property
    def db_url(self) -> str:
        """Construct database URL from components or use explicit URL."""
        if self.database_url:
            return str(self.database_url)
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def db_url_sync(self) -> str:
        """Synchronous database URL for Alembic migrations."""
        return self.db_url.replace("postgresql+asyncpg://", "postgresql://")

    # ============== Redis ==============
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_url: RedisDsn | None = None

    @property
    def redis_dsn(self) -> str:
        """Construct Redis URL from components or use explicit URL."""
        if self.redis_url:
            return str(self.redis_url)
        return f"redis://{self.redis_host}:{self.redis_port}/0"

    # ============== API ==============
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_debug: bool = False
    api_reload: bool = False
    cors_origins_str: str = Field(default="http://localhost:3000,http://localhost:8000", alias="cors_origins")

    @property
    def cors_origins(self) -> list[str]:
        """Parse CORS origins from comma-separated string."""
        return [origin.strip() for origin in self.cors_origins_str.split(",") if origin.strip()]

    # ============== LLM Configuration ==============
    google_api_key: str | None = None
    gemini_model: str = "gemini-2.5-flash"
    llm_extractor: Literal["gemini", "mock"] = "gemini"

    # ============== PubMed ==============
    ncbi_api_key: str | None = None
    ncbi_email: str = "dev@life-ai.com"

    # ============== KG Pipeline ==============
    chunk_size: int = Field(default=1000, ge=100, le=10000)
    chunk_overlap: int = Field(default=200, ge=0, le=500)
    extraction_confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    max_entities_per_chunk: int = Field(default=20, ge=1, le=100)
    max_relations_per_chunk: int = Field(default=30, ge=1, le=100)

    # ============== Celery ==============
    celery_broker_url: str | None = None
    celery_result_backend: str | None = None
    celery_task_always_eager: bool = False  # Synchronous execution for testing

    @property
    def celery_broker(self) -> str:
        """Get Celery broker URL."""
        return self.celery_broker_url or self.redis_dsn

    @property
    def celery_backend(self) -> str:
        """Get Celery result backend URL."""
        return self.celery_result_backend or self.redis_dsn

    # ============== Logging ==============
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_format: Literal["json", "console"] = "console"

    # ============== Computed Properties ==============
    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.environment == "development"

    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.environment == "production"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Global settings instance
settings = get_settings()
