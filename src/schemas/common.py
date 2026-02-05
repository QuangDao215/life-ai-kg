"""Common Pydantic schemas used across API endpoints."""

from datetime import datetime
from typing import Any, Generic, TypeVar
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

# Generic type for paginated responses
T = TypeVar("T")


# =============================================================================
# Pagination
# =============================================================================


class PaginationParams(BaseModel):
    """Query parameters for pagination."""

    page: int = Field(default=1, ge=1, description="Page number (1-indexed)")
    page_size: int = Field(
        default=20, ge=1, le=100, description="Items per page (max 100)"
    )

    @property
    def offset(self) -> int:
        """Calculate SQL offset from page number."""
        return (self.page - 1) * self.page_size

    @property
    def limit(self) -> int:
        """Alias for page_size for SQL queries."""
        return self.page_size


class PaginatedResponse(BaseModel, Generic[T]):
    """Generic paginated response wrapper."""

    items: list[T]
    total: int = Field(description="Total number of items")
    page: int = Field(description="Current page number")
    page_size: int = Field(description="Items per page")
    pages: int = Field(description="Total number of pages")

    model_config = ConfigDict(from_attributes=True)

    @classmethod
    def create(
        cls,
        items: list[T],
        total: int,
        page: int,
        page_size: int,
    ) -> "PaginatedResponse[T]":
        """Factory method to create paginated response."""
        pages = (total + page_size - 1) // page_size if page_size > 0 else 0
        return cls(
            items=items,
            total=total,
            page=page,
            page_size=page_size,
            pages=pages,
        )


# =============================================================================
# Error Responses
# =============================================================================


class ErrorDetail(BaseModel):
    """Detailed error information."""

    field: str | None = Field(default=None, description="Field that caused the error")
    message: str = Field(description="Error message")
    code: str | None = Field(default=None, description="Error code")


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str = Field(description="Error type")
    message: str = Field(description="Human-readable error message")
    details: list[ErrorDetail] | None = Field(
        default=None, description="Additional error details"
    )
    request_id: str | None = Field(default=None, description="Request ID for tracing")


class ValidationErrorResponse(BaseModel):
    """Validation error response (422)."""

    detail: list[dict[str, Any]] = Field(description="Validation error details")


# =============================================================================
# Common Fields
# =============================================================================


class TimestampMixin(BaseModel):
    """Mixin for timestamp fields."""

    created_at: datetime = Field(description="Creation timestamp")


class UUIDMixin(BaseModel):
    """Mixin for UUID primary key."""

    id: UUID = Field(description="Unique identifier")


class BaseResponse(UUIDMixin, TimestampMixin):
    """Base response model with ID and timestamps."""

    model_config = ConfigDict(from_attributes=True)


# =============================================================================
# Statistics
# =============================================================================


class CountByType(BaseModel):
    """Count grouped by type/label."""

    type: str = Field(description="Type or label name")
    count: int = Field(description="Count of items")


class DatabaseStats(BaseModel):
    """Overall database statistics."""

    documents: int = Field(description="Total documents")
    chunks: int = Field(description="Total chunks")
    chunks_processed: int = Field(description="Chunks with extractions")
    entities: int = Field(description="Total entities")
    relations: int = Field(description="Total relations")
    evidence: int = Field(description="Total evidence records")
    entities_by_type: list[CountByType] = Field(description="Entity counts by type")
    relations_by_label: list[CountByType] = Field(description="Relation counts by label")
