"""
AuditLog model for tracking record deletions and modifications.

This implements the hybrid delete strategy:
- Records are hard deleted from primary tables
- Deletions are logged to audit_log for recovery and audit trails

Key features:
- Captures full record state before deletion (old_data)
- Tracks which table and record was affected
- Optional reason field for context
- Append-only (immutable records)

Usage:
    When deleting an entity:
    1. Serialize the entity to dict
    2. Create audit log entry
    3. Delete the entity

    async with transaction(session):
        audit = AuditLog.create_deletion(
            table_name="entities",
            record_id=entity.id,
            old_data=entity.to_dict(),
            reason="Duplicate entity merged",
        )
        session.add(audit)
        await session.delete(entity)
"""

import uuid
from datetime import datetime
from enum import Enum

from sqlalchemy import DateTime, Index, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from src.db.base import Base, UUIDMixin


class AuditAction(str, Enum):
    """Types of auditable actions."""

    DELETE = "DELETE"
    UPDATE = "UPDATE"
    BULK_DELETE = "BULK_DELETE"
    MERGE = "MERGE"


class AuditLog(UUIDMixin, Base):
    """
    Audit log entry for tracking record changes and deletions.

    Attributes:
        id: UUID7 primary key
        table_name: Name of the affected table (e.g., "entities")
        record_id: UUID of the affected record
        action: Type of action (DELETE, UPDATE, etc.)
        old_data: Complete record state before the action
        new_data: Record state after action (for updates)
        reason: Optional context for why this action occurred
        created_at: When the action was logged

    This table is append-only - records should never be updated or deleted.

    Example:
        audit = AuditLog(
            table_name="entities",
            record_id=entity.id,
            action=AuditAction.DELETE,
            old_data=entity.to_dict(),
            reason="Merged duplicate entities",
        )
    """

    # === Core Fields ===
    table_name: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
        comment="Name of the affected table",
    )

    record_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        nullable=False,
        index=True,
        comment="UUID of the affected record",
    )

    action: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        index=True,
        comment="Type of action (DELETE, UPDATE, BULK_DELETE, MERGE)",
    )

    # === Data Snapshots ===
    old_data: Mapped[dict | None] = mapped_column(
        JSONB,
        nullable=True,
        comment="Complete record state before the action",
    )

    new_data: Mapped[dict | None] = mapped_column(
        JSONB,
        nullable=True,
        comment="Record state after action (for updates)",
    )

    # === Context ===
    reason: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="Optional context for why this action occurred",
    )

    # === Timestamp ===
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
        comment="When the action was logged",
    )

    def __repr__(self) -> str:
        return f"<AuditLog(table={self.table_name}, record={self.record_id}, action={self.action})>"

    # === Factory Methods ===
    @classmethod
    def create_deletion(
        cls,
        table_name: str,
        record_id: uuid.UUID,
        old_data: dict,
        reason: str | None = None,
    ) -> "AuditLog":
        """
        Create an audit log entry for a deletion.

        Args:
            table_name: Name of the table (e.g., "entities")
            record_id: UUID of the record being deleted
            old_data: Complete record state before deletion
            reason: Optional context for why deletion occurred

        Returns:
            New AuditLog instance
        """
        return cls(
            table_name=table_name,
            record_id=record_id,
            action=AuditAction.DELETE.value,
            old_data=old_data,
            reason=reason,
        )

    @classmethod
    def create_update(
        cls,
        table_name: str,
        record_id: uuid.UUID,
        old_data: dict,
        new_data: dict,
        reason: str | None = None,
    ) -> "AuditLog":
        """
        Create an audit log entry for an update.

        Args:
            table_name: Name of the table
            record_id: UUID of the record being updated
            old_data: Record state before update
            new_data: Record state after update
            reason: Optional context

        Returns:
            New AuditLog instance
        """
        return cls(
            table_name=table_name,
            record_id=record_id,
            action=AuditAction.UPDATE.value,
            old_data=old_data,
            new_data=new_data,
            reason=reason,
        )

    @classmethod
    def create_merge(
        cls,
        table_name: str,
        source_id: uuid.UUID,
        target_id: uuid.UUID,
        old_data: dict,
        reason: str | None = None,
    ) -> "AuditLog":
        """
        Create an audit log entry for a merge operation.

        Used when merging duplicate entities - the source is deleted
        and its data merged into the target.

        Args:
            table_name: Name of the table
            source_id: UUID of the record being merged (deleted)
            target_id: UUID of the record receiving the merge
            old_data: Complete state of the source record
            reason: Optional context

        Returns:
            New AuditLog instance
        """
        return cls(
            table_name=table_name,
            record_id=source_id,
            action=AuditAction.MERGE.value,
            old_data=old_data,
            new_data={"merged_into": str(target_id)},
            reason=reason,
        )

    @classmethod
    def create_bulk_deletion(
        cls,
        table_name: str,
        record_ids: list[uuid.UUID],
        reason: str | None = None,
    ) -> "AuditLog":
        """
        Create an audit log entry for a bulk deletion.

        Note: This does not capture old_data for each record.
        Use individual delete entries if you need full recovery.

        Args:
            table_name: Name of the table
            record_ids: List of UUIDs being deleted
            reason: Optional context

        Returns:
            New AuditLog instance
        """
        # Use a sentinel UUID for bulk operations
        from uuid6 import uuid7

        return cls(
            table_name=table_name,
            record_id=uuid7(),  # Generate new ID for this bulk operation
            action=AuditAction.BULK_DELETE.value,
            old_data={"deleted_ids": [str(rid) for rid in record_ids]},
            reason=reason,
        )


# === Indexes ===
# Index for finding audit logs by table
Index("ix_audit_logs_table_name", AuditLog.table_name)

# Index for finding audit logs by record
Index("ix_audit_logs_record_id", AuditLog.record_id)

# Index for finding audit logs by action type
Index("ix_audit_logs_action", AuditLog.action)

# Index for sorting by time
Index("ix_audit_logs_created_at", AuditLog.created_at.desc())

# Composite index for finding all audit entries for a specific record
Index(
    "ix_audit_logs_table_record",
    AuditLog.table_name,
    AuditLog.record_id,
)
