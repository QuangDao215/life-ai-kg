"""Unit tests for core configuration and health endpoints."""

import pytest
from fastapi.testclient import TestClient

from src.core.config import Settings, get_settings


class TestSettings:
    """Tests for application settings."""

    def test_default_settings(self) -> None:
        """Test that default settings are loaded correctly."""
        settings = Settings()
        assert settings.environment == "development"
        assert settings.postgres_user == "lifeai"
        assert settings.postgres_db == "lifeai_kg"
        assert settings.chunk_size == 1000
        assert settings.chunk_overlap == 200

    def test_db_url_construction(self) -> None:
        """Test database URL is constructed correctly from components."""
        # Create settings without loading from env file
        settings = Settings(
            _env_file=None,  # Skip loading .env file
            postgres_user="testuser",
            postgres_password="testpass",
            postgres_host="testhost",
            postgres_port=5433,
            postgres_db="testdb",
        )
        expected = "postgresql+asyncpg://testuser:testpass@testhost:5433/testdb"
        assert settings.db_url == expected

    def test_db_url_sync(self) -> None:
        """Test synchronous database URL for Alembic."""
        settings = Settings()
        assert "postgresql://" in settings.db_url_sync
        assert "asyncpg" not in settings.db_url_sync

    def test_cors_origins_from_string(self) -> None:
        """Test CORS origins parsed from comma-separated string."""
        settings = Settings(cors_origins="http://a.com, http://b.com")
        assert settings.cors_origins == ["http://a.com", "http://b.com"]

    def test_cors_origins_default(self) -> None:
        """Test CORS origins default value."""
        settings = Settings()
        assert "http://localhost:3000" in settings.cors_origins

    def test_environment_checks(self) -> None:
        """Test environment check properties."""
        dev_settings = Settings(environment="development")
        assert dev_settings.is_development is True
        assert dev_settings.is_production is False

        prod_settings = Settings(environment="production")
        assert prod_settings.is_development is False
        assert prod_settings.is_production is True

    def test_chunk_size_validation(self) -> None:
        """Test chunk size validation bounds."""
        # Valid
        settings = Settings(chunk_size=500)
        assert settings.chunk_size == 500

        # Too small - should raise
        with pytest.raises(ValueError):
            Settings(chunk_size=50)

        # Too large - should raise
        with pytest.raises(ValueError):
            Settings(chunk_size=20000)

    def test_get_settings_caching(self) -> None:
        """Test that get_settings returns cached instance."""
        settings1 = get_settings()
        settings2 = get_settings()
        assert settings1 is settings2


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    def test_health_check(self, client: TestClient) -> None:
        """Test health endpoint returns healthy status."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data

    def test_root_endpoint(self, client: TestClient) -> None:
        """Test root endpoint returns API info."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "LIFE AI Knowledge Graph API"
        assert "version" in data
        assert "docs" in data


class TestOpenAPISchema:
    """Tests for OpenAPI documentation."""

    def test_openapi_schema_available(self, client: TestClient) -> None:
        """Test that OpenAPI schema is generated."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert schema["info"]["title"] == "LIFE AI Knowledge Graph API"
        assert "paths" in schema

    def test_docs_endpoint(self, client: TestClient) -> None:
        """Test that Swagger UI docs are available."""
        response = client.get("/docs")
        assert response.status_code == 200
        assert "swagger" in response.text.lower()
