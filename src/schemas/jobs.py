"""Pydantic schemas for Job API endpoints."""

from datetime import datetime
from enum import Enum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

# =============================================================================
# Enums
# =============================================================================


class JobStatus(str, Enum):
    """Job status enumeration."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobType(str, Enum):
    """Job type enumeration."""

    KG_BUILD = "kg_build"
    DOCUMENT_CHUNK = "document_chunk"
    FULL_EXTRACTION = "full_extraction"


# =============================================================================
# Request Schemas
# =============================================================================


class KGBuildRequest(BaseModel):
    """Request to trigger KG build job."""

    document_ids: list[UUID] | None = Field(
        default=None,
        description="Specific document IDs to process (None = all unprocessed)",
    )
    limit: int | None = Field(
        default=None,
        ge=1,
        le=1000,
        description="Maximum number of chunks to process",
    )
    skip_processed: bool = Field(
        default=True,
        description="Skip chunks that already have extractions",
    )
    provider: str = Field(
        default="gemini",
        description="LLM provider to use",
    )


# =============================================================================
# Response Schemas
# =============================================================================


class JobResponse(BaseModel):
    """Schema for job response."""

    id: UUID = Field(description="Job UUID")
    job_type: JobType = Field(description="Type of job")
    status: JobStatus = Field(description="Current job status")
    progress: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Progress percentage (0-100)",
    )
    total_items: int = Field(default=0, description="Total items to process")
    processed_items: int = Field(default=0, description="Items processed so far")
    error_count: int = Field(default=0, description="Number of errors encountered")
    error_message: str | None = Field(default=None, description="Error message if failed")
    result: dict | None = Field(default=None, description="Job result data")
    created_at: datetime = Field(description="Job creation timestamp")
    started_at: datetime | None = Field(default=None, description="Job start timestamp")
    completed_at: datetime | None = Field(default=None, description="Job completion timestamp")

    model_config = ConfigDict(from_attributes=True)


class JobSummary(BaseModel):
    """Summarized job for list views."""

    id: UUID = Field(description="Job UUID")
    job_type: JobType = Field(description="Type of job")
    status: JobStatus = Field(description="Current job status")
    progress: float = Field(description="Progress percentage")
    created_at: datetime = Field(description="Job creation timestamp")
    completed_at: datetime | None = Field(description="Job completion timestamp")

    model_config = ConfigDict(from_attributes=True)


class JobCreateResponse(BaseModel):
    """Response when creating a new job."""

    id: UUID = Field(description="Job UUID")
    job_type: JobType = Field(description="Type of job")
    status: JobStatus = Field(description="Initial job status")
    message: str = Field(description="Status message")


class ExtractionProgress(BaseModel):
    """Real-time extraction progress."""

    chunks_total: int = Field(description="Total chunks to process")
    chunks_processed: int = Field(description="Chunks processed so far")
    chunks_remaining: int = Field(description="Chunks remaining")
    entities_extracted: int = Field(description="Total entities extracted")
    relations_extracted: int = Field(description="Total relations extracted")
    errors: int = Field(description="Number of errors")
    estimated_time_remaining: str | None = Field(
        default=None, description="Estimated time remaining"
    )
