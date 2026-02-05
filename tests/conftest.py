"""Pytest configuration and shared fixtures."""

import asyncio
from collections.abc import AsyncGenerator, Generator
from typing import Any

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.core.config import settings
from src.db import get_db
from src.main import app

# Configure pytest-asyncio
pytest_plugins = ("pytest_asyncio",)


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an event loop for the test session."""
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    """Create a synchronous test client for FastAPI."""
    with TestClient(app) as test_client:
        yield test_client


# Database fixtures for integration tests
@pytest_asyncio.fixture(scope="function")
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    """Create a database session for testing."""
    engine = create_async_engine(settings.db_url, echo=False)
    async_session = async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    async with async_session() as session:
        yield session

    await engine.dispose()


@pytest_asyncio.fixture(scope="function")
async def async_client(db_session: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
    """Create an asynchronous test client for FastAPI with DB override."""

    async def override_get_db() -> AsyncGenerator[AsyncSession, None]:
        yield db_session

    app.dependency_overrides[get_db] = override_get_db

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as ac:
        yield ac

    app.dependency_overrides.clear()


@pytest_asyncio.fixture(scope="function")
async def async_client_no_db() -> AsyncGenerator[AsyncClient, None]:
    """Create an async test client without DB (for non-DB endpoints)."""
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as ac:
        yield ac


@pytest.fixture
def sample_document_data() -> dict[str, Any]:
    """Sample document data for testing."""
    return {
        "pmid": "12345678",
        "title": "Ambroxol as a disease-modifying treatment for Parkinson's disease",
        "abstract": (
            "Ambroxol is a mucolytic agent that has shown potential as a treatment "
            "for Parkinson's disease through its ability to enhance GCase activity. "
            "This study examines the effects of Ambroxol on alpha-synuclein aggregation "
            "and lysosomal function in neuronal cells."
        ),
        "authors": ["Smith J", "Doe A", "Johnson B"],
        "journal": "Journal of Neurology",
        "publication_date": "2024-01-15",
    }


@pytest.fixture
def sample_extraction_result() -> dict[str, Any]:
    """Sample extraction result for testing."""
    return {
        "entities": [
            {"name": "Ambroxol", "type": "Compound/Drug", "aliases": ["ambroxol hydrochloride"]},
            {"name": "Parkinson's disease", "type": "Disease", "aliases": ["PD", "Parkinson disease"]},
            {"name": "GCase", "type": "Protein", "aliases": ["glucocerebrosidase", "GBA1"]},
            {"name": "alpha-synuclein", "type": "Protein", "aliases": ["α-synuclein", "SNCA"]},
        ],
        "relations": [
            {
                "head": "Ambroxol",
                "relation": "ACTIVATES",
                "tail": "GCase",
                "evidence_quote": "Ambroxol enhances GCase activity",
                "confidence": 0.85,
            },
            {
                "head": "alpha-synuclein",
                "relation": "ASSOCIATED_WITH",
                "tail": "Parkinson's disease",
                "evidence_quote": "alpha-synuclein aggregation in neuronal cells",
                "confidence": 0.92,
            },
        ],
    }


@pytest.fixture
def sample_entity_names() -> list[str]:
    """Sample entity names for search tests."""
    return [
        "Parkinson",
        "Ambroxol",
        "GCase",
        "alpha-synuclein",
        "lysosome",
    ]
