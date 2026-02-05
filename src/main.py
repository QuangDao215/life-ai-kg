"""FastAPI application entry point."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api import documents_router, entities_router, graph_router, jobs_router
from src.core.config import settings
from src.core.logging import setup_logging


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager for startup/shutdown events."""
    # Startup
    setup_logging()
    # TODO: Initialize database connection pool
    # TODO: Verify Redis connection
    yield
    # Shutdown
    # TODO: Close database connections
    # TODO: Close Redis connections


def create_app() -> FastAPI:
    """Application factory for creating the FastAPI instance."""
    app = FastAPI(
        title="LIFE AI Knowledge Graph API",
        description=(
            "Backend service for building and serving a biomedical Knowledge Graph "
            "focused on Ambroxol-Parkinson's disease relationships.\n\n"
            "## Features\n"
            "- **Documents**: Manage PubMed publications\n"
            "- **Jobs**: Trigger and monitor KG extraction jobs\n"
            "- **Entities**: Search and explore extracted entities\n"
            "- **Graph**: Query subgraphs and find paths between entities\n"
        ),
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routers
    app.include_router(
        documents_router,
        prefix="/api/v1/documents",
        tags=["Documents"],
    )
    app.include_router(
        jobs_router,
        prefix="/api/v1/jobs",
        tags=["Jobs"],
    )
    app.include_router(
        entities_router,
        prefix="/api/v1/entities",
        tags=["Entities"],
    )
    app.include_router(
        graph_router,
        prefix="/api/v1/graph",
        tags=["Graph"],
    )

    @app.get("/health", tags=["Health"])
    async def health_check() -> dict[str, str]:
        """Health check endpoint for container orchestration."""
        return {"status": "healthy", "version": "0.1.0"}

    @app.get("/", tags=["Root"])
    async def root() -> dict[str, str]:
        """Root endpoint with API information."""
        return {
            "service": "LIFE AI Knowledge Graph API",
            "version": "0.1.0",
            "docs": "/docs",
            "health": "/health",
        }

    return app


# Create the app instance
app = create_app()

