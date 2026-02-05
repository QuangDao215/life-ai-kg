"""
Job model for tracking async pipeline status and progress.

Jobs represent asynchronous operations like:
- KG build (chunking, extraction, normalization, storage)
- Document ingestion batches
- Evaluation runs

Key features:
- Status tracking (pending, running, completed, failed, cancelled)
- Progress percentage and item counts
- Error capture with diagnostic context (REQ-KG-07)
- Timestamps for duration analysis

Requirements addressed:
- REQ-KG-06: Job status persistence with progress
- REQ-KG-07: Failure diagnostics (stage, document, error context)
"""

from datetime import datetime

from sqlalchemy import CheckConstraint, DateTime, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from src.db.base import Base, UUIDMixin
from src.db.enums import JobStatus


class Job(UUIDMixin, Base):
    """
    An async job tracking pipeline status and progress.

    Attributes:
        id: UUID7 primary key (also serves as job identifier)
        job_type: Type of job ("kg_build", "document_ingest", etc.)
        status: Current state (pending, running, completed, failed, cancelled)
        progress: Percentage complete (0-100)
        total_items: Total items to process
        processed_items: Items processed so far
        error: Error message if failed
        error_context: Diagnostic details (stage, document_id, stack trace)
        extra_data: Job parameters and configuration
        created_at: When job was created
        started_at: When processing began
        completed_at: When processing finished

    Lifecycle:
        PENDING -> RUNNING -> COMPLETED
                          |-> FAILED
                          |-> CANCELLED

    Example:
        job = Job(
            job_type="kg_build",
            status=JobStatus.PENDING,
            extra_data={"document_ids": [...], "chunk_size": 1000},
        )

        # Start processing
        job.status = JobStatus.RUNNING
        job.started_at = datetime.utcnow()

        # Update progress
        job.processed_items = 10
        job.progress = 50

        # Complete
        job.status = JobStatus.COMPLETED
        job.completed_at = datetime.utcnow()
        job.progress = 100
    """

    # === Core Fields ===
    job_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
        comment="Type of job (kg_build, document_ingest, evaluation)",
    )

    status: Mapped[JobStatus] = mapped_column(
        nullable=False,
        default=JobStatus.PENDING,
        index=True,
        comment="Current job state",
    )

    # === Progress Tracking ===
    progress: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="Percentage complete (0-100)",
    )

    total_items: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="Total items to process",
    )

    processed_items: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="Items processed so far",
    )

    # === Error Tracking (REQ-KG-07) ===
    error: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="Error message if failed",
    )

    error_context: Mapped[dict | None] = mapped_column(
        JSONB,
        nullable=True,
        comment="Diagnostic details: {stage, document_id, chunk_id, stack_trace}",
    )

    # === Job Configuration ===
    extra_data: Mapped[dict | None] = mapped_column(
        JSONB,
        nullable=True,
        default=dict,
        comment="Job parameters and configuration",
    )

    # === Timestamps ===
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
        comment="When job was created",
    )

    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="When processing began",
    )

    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="When processing finished (success or failure)",
    )

    # === Table Constraints ===
    __table_args__ = (
        # Progress must be between 0 and 100
        CheckConstraint(
            "progress >= 0 AND progress <= 100",
            name="ck_jobs_progress_range",
        ),
        # processed_items cannot exceed total_items if total is set
        CheckConstraint(
            "(total_items IS NULL) OR (processed_items <= total_items)",
            name="ck_jobs_processed_not_exceeds_total",
        ),
    )

    def __repr__(self) -> str:
        return f"<Job(id={self.id}, type={self.job_type}, status={self.status.value}, progress={self.progress}%)>"

    # === Status Management ===
    def start(self) -> None:
        """Mark job as started."""
        self.status = JobStatus.RUNNING
        self.started_at = datetime.utcnow()

    def complete(self) -> None:
        """Mark job as successfully completed."""
        self.status = JobStatus.COMPLETED
        self.completed_at = datetime.utcnow()
        self.progress = 100

    def fail(
        self,
        error: str,
        stage: str | None = None,
        document_id: str | None = None,
        chunk_id: str | None = None,
        stack_trace: str | None = None,
    ) -> None:
        """
        Mark job as failed with error details.

        Args:
            error: Human-readable error message
            stage: Pipeline stage where failure occurred
            document_id: ID of document being processed
            chunk_id: ID of chunk being processed
            stack_trace: Full stack trace for debugging
        """
        self.status = JobStatus.FAILED
        self.completed_at = datetime.utcnow()
        self.error = error
        self.error_context = {
            "stage": stage,
            "document_id": document_id,
            "chunk_id": chunk_id,
            "stack_trace": stack_trace,
        }

    def cancel(self) -> None:
        """Mark job as cancelled."""
        self.status = JobStatus.CANCELLED
        self.completed_at = datetime.utcnow()

    # === Progress Management ===
    def update_progress(self, processed: int, total: int | None = None) -> None:
        """
        Update progress counters and percentage.

        Args:
            processed: Number of items processed
            total: Total items (optional, updates if provided)
        """
        self.processed_items = processed
        if total is not None:
            self.total_items = total

        if self.total_items and self.total_items > 0:
            self.progress = min(100, int((processed / self.total_items) * 100))

    def increment_progress(self, count: int = 1) -> None:
        """Increment processed items by count."""
        self.processed_items += count
        if self.total_items and self.total_items > 0:
            self.progress = min(100, int((self.processed_items / self.total_items) * 100))

    # === Properties ===
    @property
    def is_terminal(self) -> bool:
        """Check if job is in a terminal state."""
        return self.status.is_terminal

    @property
    def is_running(self) -> bool:
        """Check if job is currently running."""
        return self.status == JobStatus.RUNNING

    @property
    def is_successful(self) -> bool:
        """Check if job completed successfully."""
        return self.status == JobStatus.COMPLETED

    @property
    def duration_seconds(self) -> float | None:
        """
        Calculate job duration in seconds.

        Returns None if job hasn't started or completed.
        """
        if not self.started_at:
            return None
        end_time = self.completed_at or datetime.utcnow()
        return (end_time - self.started_at).total_seconds()

    @property
    def items_remaining(self) -> int | None:
        """Calculate remaining items to process."""
        if self.total_items is None:
            return None
        return max(0, self.total_items - self.processed_items)


# === Indexes ===
# Index for finding jobs by status
Index("ix_jobs_status", Job.status)

# Index for finding jobs by type
Index("ix_jobs_job_type", Job.job_type)

# Composite index for filtering by type and status
Index("ix_jobs_type_status", Job.job_type, Job.status)

# Index for sorting by creation time
Index("ix_jobs_created_at", Job.created_at.desc())
