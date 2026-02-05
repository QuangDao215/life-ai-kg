"""Pydantic schemas for Entity API endpoints."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

# =============================================================================
# Request Schemas
# =============================================================================


class EntitySearchParams(BaseModel):
    """Search parameters for entities."""

    query: str = Field(
        min_length=1,
        max_length=200,
        description="Search query (name or alias)",
    )
    entity_type: str | None = Field(
        default=None,
        description="Filter by entity type",
    )
    page: int = Field(default=1, ge=1, description="Page number")
    page_size: int = Field(default=20, ge=1, le=100, description="Items per page")


# =============================================================================
# Response Schemas
# =============================================================================


class EntityResponse(BaseModel):
    """Schema for entity response."""

    id: UUID = Field(description="Entity UUID")
    name: str = Field(description="Entity name")
    normalized_name: str = Field(description="Normalized entity name")
    type: str = Field(description="Entity type")
    description: str | None = Field(default=None, description="Entity description")
    aliases: list[str] = Field(default_factory=list, description="Alternative names")
    external_ids: dict = Field(default_factory=dict, description="External database IDs")
    extra_data: dict = Field(default_factory=dict, description="Additional metadata")
    created_at: datetime = Field(description="Creation timestamp")
    updated_at: datetime = Field(description="Last update timestamp")

    model_config = ConfigDict(from_attributes=True)


class EntitySummary(BaseModel):
    """Summarized entity for list views."""

    id: UUID = Field(description="Entity UUID")
    name: str = Field(description="Entity name")
    type: str = Field(description="Entity type")
    relation_count: int = Field(default=0, description="Number of relations")

    model_config = ConfigDict(from_attributes=True)


class RelationResponse(BaseModel):
    """Schema for relation response."""

    id: UUID = Field(description="Relation UUID")
    head_id: UUID = Field(description="Head entity UUID")
    tail_id: UUID = Field(description="Tail entity UUID")
    label: str = Field(description="Relation label/type")
    confidence: float = Field(description="Confidence score (0-1)")
    created_at: datetime = Field(description="Creation timestamp")

    model_config = ConfigDict(from_attributes=True)


class RelationWithEntities(RelationResponse):
    """Relation with head and tail entity details."""

    head: EntitySummary = Field(description="Head entity")
    tail: EntitySummary = Field(description="Tail entity")


class EvidenceResponse(BaseModel):
    """Schema for evidence response."""

    id: UUID = Field(description="Evidence UUID")
    relation_id: UUID = Field(description="Associated relation UUID")
    chunk_id: UUID = Field(description="Source chunk UUID")
    quote: str = Field(description="Evidence quote text")
    quote_start: int | None = Field(description="Quote start offset in chunk")
    quote_end: int | None = Field(description="Quote end offset in chunk")
    created_at: datetime = Field(description="Creation timestamp")

    # Optional: include source document info
    document_pmid: str | None = Field(default=None, description="Source document PMID")
    document_title: str | None = Field(default=None, description="Source document title")

    model_config = ConfigDict(from_attributes=True)


class RelationWithEvidence(RelationWithEntities):
    """Relation with entities and evidence."""

    evidence: list[EvidenceResponse] = Field(
        default_factory=list, description="Supporting evidence"
    )


class NeighborRelation(BaseModel):
    """A relation in the entity neighborhood."""

    relation_id: UUID = Field(description="Relation UUID")
    direction: str = Field(description="'outgoing' or 'incoming'")
    label: str = Field(description="Relation label")
    confidence: float = Field(description="Confidence score")
    neighbor: EntitySummary = Field(description="The connected entity")
    evidence_count: int = Field(default=0, description="Number of evidence records")

    model_config = ConfigDict(from_attributes=True)


class EntityNeighborhood(BaseModel):
    """Entity with its neighborhood (adjacent nodes and relations)."""

    entity: EntityResponse = Field(description="The entity")
    relations: list[NeighborRelation] = Field(
        default_factory=list, description="Connected relations"
    )
    total_relations: int = Field(description="Total number of relations")
    relation_counts_by_label: dict[str, int] = Field(
        default_factory=dict, description="Count of relations by label"
    )
    neighbor_counts_by_type: dict[str, int] = Field(
        default_factory=dict, description="Count of neighbors by entity type"
    )


class EntityTypeCount(BaseModel):
    """Count of entities by type."""

    type: str = Field(description="Entity type")
    count: int = Field(description="Number of entities")
