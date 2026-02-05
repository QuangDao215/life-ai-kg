"""Pydantic schemas for API request/response models."""

from src.schemas.common import (
    CountByType,
    DatabaseStats,
    ErrorDetail,
    ErrorResponse,
    PaginatedResponse,
    PaginationParams,
    ValidationErrorResponse,
)
from src.schemas.documents import (
    ChunkResponse,
    DocumentCreate,
    DocumentResponse,
    DocumentSummary,
    DocumentWithChunks,
)
from src.schemas.entities import (
    EntityNeighborhood,
    EntityResponse,
    EntitySearchParams,
    EntitySummary,
    EntityTypeCount,
    EvidenceResponse,
    NeighborRelation,
    RelationResponse,
    RelationWithEntities,
    RelationWithEvidence,
)
from src.schemas.graph import (
    GraphEdge,
    GraphEdgeWithEvidence,
    GraphNode,
    PathParams,
    PathResponse,
    PathsResponse,
    PathStep,
    SubgraphParams,
    SubgraphResponse,
    SubgraphStats,
)
from src.schemas.jobs import (
    ExtractionProgress,
    JobCreateResponse,
    JobResponse,
    JobStatus,
    JobSummary,
    JobType,
    KGBuildRequest,
)

__all__ = [
    # Common
    "CountByType",
    "DatabaseStats",
    "ErrorDetail",
    "ErrorResponse",
    "PaginatedResponse",
    "PaginationParams",
    "ValidationErrorResponse",
    # Documents
    "ChunkResponse",
    "DocumentCreate",
    "DocumentResponse",
    "DocumentSummary",
    "DocumentWithChunks",
    # Entities
    "EntityNeighborhood",
    "EntityResponse",
    "EntitySearchParams",
    "EntitySummary",
    "EntityTypeCount",
    "EvidenceResponse",
    "NeighborRelation",
    "RelationResponse",
    "RelationWithEntities",
    "RelationWithEvidence",
    # Graph
    "GraphEdge",
    "GraphEdgeWithEvidence",
    "GraphNode",
    "PathParams",
    "PathResponse",
    "PathsResponse",
    "PathStep",
    "SubgraphParams",
    "SubgraphResponse",
    "SubgraphStats",
    # Jobs
    "ExtractionProgress",
    "JobCreateResponse",
    "JobResponse",
    "JobStatus",
    "JobSummary",
    "JobType",
    "KGBuildRequest",
]
