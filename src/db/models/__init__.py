"""
Database models for the Knowledge Graph.

This package contains SQLAlchemy models for:
- Document: Source publications from PubMed
- Chunk: Text segments for LLM extraction
- Entity: KG nodes (diseases, genes, compounds, etc.)
- Relation: KG edges (associations, interactions, etc.)
- Evidence: Provenance linking relations to source text
- Job: Async pipeline status tracking
- AuditLog: Record deletion/modification tracking

Usage:
    from src.db.models import Document, Entity, Relation
    from src.db.models import EntityType, RelationLabel, JobStatus

All models inherit from the base classes in src.db.base and use:
- UUID7 primary keys (time-sortable, globally unique)
- Timestamp mixins (created_at, updated_at)
- JSONB for flexible metadata
- Proper indexes for query performance
"""

from src.db.models.audit_log import AuditAction, AuditLog
from src.db.models.chunk import Chunk
from src.db.models.document import Document
from src.db.models.entity import Entity
from src.db.models.evidence import Evidence
from src.db.models.job import Job
from src.db.models.relation import Relation

__all__ = [
    # Core KG models
    "Document",
    "Chunk",
    "Entity",
    "Relation",
    "Evidence",
    # Pipeline models
    "Job",
    # Audit models
    "AuditLog",
    "AuditAction",
]
