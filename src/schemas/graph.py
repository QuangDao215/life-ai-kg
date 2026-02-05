"""Pydantic schemas for Graph API endpoints."""

from uuid import UUID

from pydantic import BaseModel, Field

from src.schemas.entities import (
    EntitySummary,
    EvidenceResponse,
    RelationWithEntities,
)

# =============================================================================
# Request Schemas
# =============================================================================


class SubgraphParams(BaseModel):
    """Parameters for subgraph retrieval."""

    entity_id: UUID = Field(description="Starting entity UUID")
    depth: int = Field(
        default=1,
        ge=1,
        le=3,
        description="Maximum traversal depth (1-3)",
    )
    max_nodes: int = Field(
        default=50,
        ge=1,
        le=200,
        description="Maximum number of nodes to return",
    )
    max_edges: int = Field(
        default=100,
        ge=1,
        le=500,
        description="Maximum number of edges to return",
    )
    relation_types: list[str] | None = Field(
        default=None,
        description="Filter by relation types (None = all)",
    )
    entity_types: list[str] | None = Field(
        default=None,
        description="Filter by entity types (None = all)",
    )
    min_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum relation confidence",
    )
    include_evidence: bool = Field(
        default=False,
        description="Include evidence for relations",
    )


class PathParams(BaseModel):
    """Parameters for path computation."""

    source_id: UUID = Field(description="Source entity UUID")
    target_id: UUID = Field(description="Target entity UUID")
    max_hops: int = Field(
        default=3,
        ge=1,
        le=5,
        description="Maximum path length (hops)",
    )
    relation_types: list[str] | None = Field(
        default=None,
        description="Filter by relation types (None = all)",
    )
    min_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum relation confidence",
    )
    include_evidence: bool = Field(
        default=False,
        description="Include evidence for relations",
    )


# =============================================================================
# Response Schemas
# =============================================================================


class GraphNode(BaseModel):
    """Node in graph response."""

    id: UUID = Field(description="Entity UUID")
    name: str = Field(description="Entity name")
    type: str = Field(description="Entity type")
    depth: int = Field(
        default=0,
        description="Distance from starting node (for subgraph)",
    )

    class Config:
        from_attributes = True


class GraphEdge(BaseModel):
    """Edge in graph response."""

    id: UUID = Field(description="Relation UUID")
    source: UUID = Field(description="Source entity UUID")
    target: UUID = Field(description="Target entity UUID")
    label: str = Field(description="Relation label")
    confidence: float = Field(description="Confidence score")
    evidence_count: int = Field(default=0, description="Number of evidence records")

    class Config:
        from_attributes = True


class GraphEdgeWithEvidence(GraphEdge):
    """Edge with evidence details."""

    evidence: list[EvidenceResponse] = Field(
        default_factory=list, description="Supporting evidence"
    )


class SubgraphResponse(BaseModel):
    """Response for subgraph retrieval."""

    nodes: list[GraphNode] = Field(description="Nodes in subgraph")
    edges: list[GraphEdge | GraphEdgeWithEvidence] = Field(
        description="Edges in subgraph"
    )
    center_node: UUID = Field(description="Starting node UUID")
    depth: int = Field(description="Maximum depth reached")
    truncated: bool = Field(
        default=False,
        description="Whether results were truncated due to limits",
    )
    stats: "SubgraphStats" = Field(description="Subgraph statistics")


class SubgraphStats(BaseModel):
    """Statistics for subgraph response."""

    node_count: int = Field(description="Number of nodes")
    edge_count: int = Field(description="Number of edges")
    nodes_by_type: dict[str, int] = Field(description="Node counts by entity type")
    edges_by_label: dict[str, int] = Field(description="Edge counts by relation label")


# Update forward reference
SubgraphResponse.model_rebuild()


class PathStep(BaseModel):
    """A step in a path between entities."""

    entity: EntitySummary = Field(description="Entity at this step")
    relation: RelationWithEntities | None = Field(
        default=None,
        description="Relation to next entity (None for last step)",
    )
    direction: str | None = Field(
        default=None,
        description="'forward' or 'backward' (relation direction)",
    )
    evidence: list[EvidenceResponse] = Field(
        default_factory=list,
        description="Evidence for this step's relation",
    )


class PathResponse(BaseModel):
    """A path between two entities."""

    source: EntitySummary = Field(description="Source entity")
    target: EntitySummary = Field(description="Target entity")
    path_length: int = Field(description="Number of hops in path")
    steps: list[PathStep] = Field(description="Steps in the path")
    total_confidence: float = Field(
        description="Product of relation confidences along path"
    )


class PathsResponse(BaseModel):
    """Response for path computation (may include multiple paths)."""

    paths: list[PathResponse] = Field(description="Found paths")
    source: EntitySummary = Field(description="Source entity")
    target: EntitySummary = Field(description="Target entity")
    path_count: int = Field(description="Number of paths found")
    shortest_path_length: int | None = Field(
        default=None, description="Length of shortest path"
    )
    searched_depth: int = Field(description="Maximum depth searched")
