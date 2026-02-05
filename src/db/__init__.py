"""
Database package - SQLAlchemy models, session management, and utilities.

Exports:
- Base classes and mixins for model definition
- Session management for FastAPI and standalone usage
- Database lifecycle utilities
- Controlled vocabulary enums
- All database models

Usage:
    from src.db import Base, get_db, get_db_context
    from src.db import Document, Entity, Relation, Evidence
    from src.db import EntityType, RelationLabel, JobStatus
"""

from src.db.base import (
    # Engine and session factory
    AsyncSessionLocal,
    # Base classes
    Base,
    # Mixins
    CreatedAtMixin,
    TimestampMixin,
    UUIDBase,
    UUIDCreatedBase,
    UUIDMixin,
    UUIDTimestampBase,
    # Lifecycle utilities
    dispose_engine,
    drop_db,
    engine,
    init_db,
    metadata,
)
from src.db.enums import (
    # Enums
    EntityType,
    JobStatus,
    RelationLabel,
    # Validation helpers
    is_valid_entity_type,
    is_valid_relation_label,
    validate_entity_type,
    validate_relation_label,
)
from src.db.models import (
    # Audit
    AuditAction,
    AuditLog,
    # Core models
    Chunk,
    Document,
    Entity,
    Evidence,
    Job,
    Relation,
)
from src.db.session import get_db, get_db_context, transaction

__all__ = [
    # Base classes
    "Base",
    "UUIDBase",
    "UUIDCreatedBase",
    "UUIDTimestampBase",
    # Mixins
    "UUIDMixin",
    "TimestampMixin",
    "CreatedAtMixin",
    # Enums
    "EntityType",
    "RelationLabel",
    "JobStatus",
    # Validation helpers
    "is_valid_entity_type",
    "is_valid_relation_label",
    "validate_entity_type",
    "validate_relation_label",
    # Models
    "Document",
    "Chunk",
    "Entity",
    "Relation",
    "Evidence",
    "Job",
    "AuditLog",
    "AuditAction",
    # Engine and factory
    "engine",
    "AsyncSessionLocal",
    "metadata",
    # Session utilities
    "get_db",
    "get_db_context",
    "transaction",
    # Lifecycle
    "init_db",
    "drop_db",
    "dispose_engine",
]
