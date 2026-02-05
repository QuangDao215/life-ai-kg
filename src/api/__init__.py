"""API routers for the LIFE AI Knowledge Graph service."""

from src.api.documents import router as documents_router
from src.api.entities import router as entities_router
from src.api.graph import router as graph_router
from src.api.jobs import router as jobs_router

__all__ = [
    "documents_router",
    "entities_router",
    "graph_router",
    "jobs_router",
]
