"""Pydantic schemas for Document API endpoints."""

from datetime import date, datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

# =============================================================================
# Request Schemas
# =============================================================================


class DocumentCreate(BaseModel):
    """Schema for creating a new document."""

    pmid: str = Field(description="PubMed ID", examples=["12345678"])
    title: str = Field(description="Document title")
    abstract: str | None = Field(default=None, description="Document abstract text")
    authors: list[dict] | None = Field(default=None, description="List of author objects")
    journal: str | None = Field(default=None, description="Journal name")
    publication_date: date | None = Field(default=None, description="Publication date")
    doi: str | None = Field(default=None, description="Digital Object Identifier")
    mesh_terms: list[str] | None = Field(default=None, description="MeSH terms")
    extra_data: dict | None = Field(default=None, description="Additional metadata")


# =============================================================================
# Response Schemas
# =============================================================================


class DocumentResponse(BaseModel):
    """Schema for document response."""

    id: UUID = Field(description="Document UUID")
    pmid: str = Field(description="PubMed ID")
    title: str = Field(description="Document title")
    abstract: str | None = Field(description="Document abstract text")
    authors: list[dict] | None = Field(description="List of author objects")
    journal: str | None = Field(description="Journal name")
    publication_date: date | None = Field(description="Publication date")
    doi: str | None = Field(description="Digital Object Identifier")
    mesh_terms: list[str] | None = Field(default=None, description="MeSH terms")
    extra_data: dict | None = Field(description="Additional metadata")
    created_at: datetime = Field(description="Creation timestamp")

    # Computed fields
    chunk_count: int | None = Field(default=None, description="Number of chunks")
    has_extractions: bool | None = Field(
        default=None, description="Whether document has been processed"
    )

    model_config = ConfigDict(from_attributes=True)


class DocumentSummary(BaseModel):
    """Summarized document for list views."""

    id: UUID = Field(description="Document UUID")
    pmid: str = Field(description="PubMed ID")
    title: str = Field(description="Document title")
    journal: str | None = Field(description="Journal name")
    publication_date: date | None = Field(description="Publication date")
    chunk_count: int = Field(default=0, description="Number of chunks")
    has_extractions: bool = Field(
        default=False, description="Whether document has been processed"
    )
    created_at: datetime = Field(description="Creation timestamp")

    model_config = ConfigDict(from_attributes=True)


class DocumentWithChunks(DocumentResponse):
    """Document with its chunks included."""

    chunks: list["ChunkResponse"] = Field(description="Document chunks")


class ChunkResponse(BaseModel):
    """Schema for chunk response."""

    id: UUID = Field(description="Chunk UUID")
    chunk_index: int = Field(description="Chunk index within document")
    text: str = Field(description="Chunk text content")
    start_offset: int = Field(description="Start character offset")
    end_offset: int = Field(description="End character offset")
    token_count: int | None = Field(description="Estimated token count")
    has_extractions: bool = Field(
        default=False, description="Whether chunk has evidence records"
    )
    created_at: datetime = Field(description="Creation timestamp")

    model_config = ConfigDict(from_attributes=True)


# Update forward reference
DocumentWithChunks.model_rebuild()
