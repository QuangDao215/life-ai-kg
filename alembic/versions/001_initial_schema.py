"""Initial schema - create all KG tables.

Revision ID: 001_initial_schema
Revises:
Create Date: 2024-01-15

This migration creates the complete Knowledge Graph schema:
- documents: PubMed publications
- chunks: Text segments for LLM extraction
- entities: KG nodes (diseases, genes, proteins, etc.)
- relations: KG edges (associations, interactions, etc.)
- evidence: Provenance linking relations to source text
- jobs: Async pipeline status tracking
- audit_logs: Record deletion/modification tracking

It also creates:
- ENUM types for entity_type, relation_label, job_status
- All indexes for query performance
- All constraints for data integrity
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "001_initial_schema"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create all tables, enums, indexes, and constraints."""

    # ==========================================================================
    # Create ENUM types
    # ==========================================================================

    # EntityType enum
    entity_type_enum = postgresql.ENUM(
        "Disease",
        "Gene",
        "Protein",
        "Pathway",
        "CellType",
        "Tissue/Region",
        "Compound/Drug",
        "Phenotype",
        "Biomarker",
        name="entitytype",
        create_type=False,  # We create explicitly below
    )
    entity_type_enum.create(op.get_bind(), checkfirst=True)

    # RelationLabel enum
    relation_label_enum = postgresql.ENUM(
        "ASSOCIATED_WITH",
        "INCREASES_RISK",
        "DECREASES_RISK",
        "UPREGULATES",
        "DOWNREGULATES",
        "ACTIVATES",
        "INHIBITS",
        "BINDS",
        "BIOMARKER_FOR",
        "PART_OF",
        "EXPRESSED_IN",
        name="relationlabel",
        create_type=False,  # We create explicitly below
    )
    relation_label_enum.create(op.get_bind(), checkfirst=True)

    # JobStatus enum
    job_status_enum = postgresql.ENUM(
        "pending",
        "running",
        "completed",
        "failed",
        "cancelled",
        name="jobstatus",
        create_type=False,  # We create explicitly below
    )
    job_status_enum.create(op.get_bind(), checkfirst=True)

    # ==========================================================================
    # Create tables
    # ==========================================================================

    # --------------------------------------------------------------------------
    # documents table
    # --------------------------------------------------------------------------
    op.create_table(
        "documents",
        sa.Column("id", sa.UUID(), nullable=False, comment="UUID7 primary key"),
        sa.Column(
            "pmid",
            sa.String(length=20),
            nullable=False,
            comment="PubMed ID - unique identifier for idempotent ingestion",
        ),
        sa.Column("title", sa.Text(), nullable=False, comment="Article title"),
        sa.Column(
            "abstract",
            sa.Text(),
            nullable=True,
            comment="Article abstract - main source for entity extraction",
        ),
        sa.Column(
            "authors",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
            comment="List of authors [{name, affiliation}]",
        ),
        sa.Column(
            "journal", sa.String(length=500), nullable=True, comment="Journal name"
        ),
        sa.Column(
            "publication_date", sa.Date(), nullable=True, comment="Publication date"
        ),
        sa.Column(
            "doi",
            sa.String(length=100),
            nullable=True,
            comment="Digital Object Identifier",
        ),
        sa.Column(
            "mesh_terms",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
            comment="MeSH terms assigned to this article",
        ),
        sa.Column(
            "extra_data",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
            comment="Additional metadata (keywords, affiliations, etc.)",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
            comment="When the document was ingested",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_documents"),
        sa.UniqueConstraint("pmid", name="uq_documents_pmid"),
    )
    op.create_index("ix_documents_pmid", "documents", ["pmid"], unique=False)
    op.create_index(
        "ix_documents_publication_date", "documents", ["publication_date"], unique=False
    )

    # --------------------------------------------------------------------------
    # chunks table
    # --------------------------------------------------------------------------
    op.create_table(
        "chunks",
        sa.Column("id", sa.UUID(), nullable=False, comment="UUID7 primary key"),
        sa.Column(
            "document_id", sa.UUID(), nullable=False, comment="Parent document"
        ),
        sa.Column(
            "chunk_index",
            sa.Integer(),
            nullable=False,
            comment="Position in document sequence (0-indexed)",
        ),
        sa.Column("text", sa.Text(), nullable=False, comment="Chunk content"),
        sa.Column(
            "start_offset",
            sa.Integer(),
            nullable=False,
            comment="Character start position in original document",
        ),
        sa.Column(
            "end_offset",
            sa.Integer(),
            nullable=False,
            comment="Character end position in original document",
        ),
        sa.Column(
            "token_count",
            sa.Integer(),
            nullable=True,
            comment="Approximate token count for LLM context management",
        ),
        sa.Column(
            "extra_data",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
            comment="Chunking parameters (size, overlap, method)",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
            comment="When the chunk was created",
        ),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["documents.id"],
            name="fk_chunks_document_id_documents",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_chunks"),
        sa.UniqueConstraint(
            "document_id", "chunk_index", name="uq_chunks_document_id_chunk_index"
        ),
    )
    op.create_index("ix_chunks_document_id", "chunks", ["document_id"], unique=False)
    op.create_index(
        "ix_chunks_document_id_chunk_index",
        "chunks",
        ["document_id", "chunk_index"],
        unique=False,
    )

    # --------------------------------------------------------------------------
    # entities table
    # --------------------------------------------------------------------------
    op.create_table(
        "entities",
        sa.Column("id", sa.UUID(), nullable=False, comment="UUID7 primary key"),
        sa.Column(
            "name",
            sa.String(length=500),
            nullable=False,
            comment="Display name (original form)",
        ),
        sa.Column(
            "normalized_name",
            sa.String(length=500),
            nullable=False,
            comment="Canonical form for matching (lowercased, trimmed)",
        ),
        sa.Column(
            "type",
            entity_type_enum,
            nullable=False,
            comment="Entity type from controlled vocabulary",
        ),
        sa.Column(
            "description", sa.Text(), nullable=True, comment="Description of the entity"
        ),
        sa.Column(
            "aliases",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
            comment="Alternative names/synonyms",
        ),
        sa.Column(
            "external_ids",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
            comment="External identifiers {source: id} e.g., {umls: C0030567}",
        ),
        sa.Column(
            "extra_data",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
            comment="Additional attributes",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
            comment="When first extracted",
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
            comment="Last modification time",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_entities"),
    )
    op.create_index("ix_entities_name", "entities", ["name"], unique=False)
    op.create_index(
        "ix_entities_normalized_name", "entities", ["normalized_name"], unique=False
    )
    op.create_index("ix_entities_type", "entities", ["type"], unique=False)
    op.create_index(
        "uq_entities_normalized_name_type",
        "entities",
        ["normalized_name", "type"],
        unique=True,
    )

    # --------------------------------------------------------------------------
    # relations table
    # --------------------------------------------------------------------------
    op.create_table(
        "relations",
        sa.Column("id", sa.UUID(), nullable=False, comment="UUID7 primary key"),
        sa.Column(
            "head_id", sa.UUID(), nullable=False, comment="Source entity (subject)"
        ),
        sa.Column(
            "tail_id", sa.UUID(), nullable=False, comment="Target entity (object)"
        ),
        sa.Column(
            "label",
            relation_label_enum,
            nullable=False,
            comment="Relation type from controlled vocabulary",
        ),
        sa.Column(
            "confidence",
            sa.Float(),
            nullable=True,
            comment="Extraction confidence score (0.0 to 1.0)",
        ),
        sa.Column(
            "extra_data",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
            comment="Additional attributes (extraction method, etc.)",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
            comment="When first extracted",
        ),
        sa.CheckConstraint(
            "(confidence IS NULL) OR (confidence >= 0.0 AND confidence <= 1.0)",
            name="ck_relations_confidence_range",
        ),
        sa.ForeignKeyConstraint(
            ["head_id"],
            ["entities.id"],
            name="fk_relations_head_id_entities",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["tail_id"],
            ["entities.id"],
            name="fk_relations_tail_id_entities",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_relations"),
    )
    op.create_index("ix_relations_head_id", "relations", ["head_id"], unique=False)
    op.create_index("ix_relations_tail_id", "relations", ["tail_id"], unique=False)
    op.create_index("ix_relations_label", "relations", ["label"], unique=False)
    op.create_index(
        "ix_relations_head_tail", "relations", ["head_id", "tail_id"], unique=False
    )
    op.create_index(
        "uq_relations_head_tail_label",
        "relations",
        ["head_id", "tail_id", "label"],
        unique=True,
    )

    # --------------------------------------------------------------------------
    # evidence table
    # --------------------------------------------------------------------------
    op.create_table(
        "evidence",
        sa.Column("id", sa.UUID(), nullable=False, comment="UUID7 primary key"),
        sa.Column(
            "relation_id",
            sa.UUID(),
            nullable=False,
            comment="The relation this evidence supports",
        ),
        sa.Column(
            "chunk_id",
            sa.UUID(),
            nullable=False,
            comment="The source chunk containing this evidence",
        ),
        sa.Column(
            "quote",
            sa.Text(),
            nullable=False,
            comment="Evidence text supporting the relation (REQUIRED)",
        ),
        sa.Column(
            "quote_start",
            sa.Integer(),
            nullable=True,
            comment="Character start offset within chunk",
        ),
        sa.Column(
            "quote_end",
            sa.Integer(),
            nullable=True,
            comment="Character end offset within chunk",
        ),
        sa.Column(
            "quote_hash",
            sa.String(length=64),
            nullable=False,
            comment="SHA256 hash of quote for deduplication",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
            comment="When the evidence was extracted",
        ),
        sa.ForeignKeyConstraint(
            ["chunk_id"],
            ["chunks.id"],
            name="fk_evidence_chunk_id_chunks",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["relation_id"],
            ["relations.id"],
            name="fk_evidence_relation_id_relations",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_evidence"),
        sa.UniqueConstraint(
            "relation_id",
            "chunk_id",
            "quote_hash",
            name="uq_evidence_relation_chunk_quote",
        ),
    )
    op.create_index("ix_evidence_relation_id", "evidence", ["relation_id"], unique=False)
    op.create_index("ix_evidence_chunk_id", "evidence", ["chunk_id"], unique=False)
    op.create_index("ix_evidence_quote_hash", "evidence", ["quote_hash"], unique=False)

    # --------------------------------------------------------------------------
    # jobs table
    # --------------------------------------------------------------------------
    op.create_table(
        "jobs",
        sa.Column("id", sa.UUID(), nullable=False, comment="UUID7 primary key"),
        sa.Column(
            "job_type",
            sa.String(length=50),
            nullable=False,
            comment="Type of job (kg_build, document_ingest, evaluation)",
        ),
        sa.Column(
            "status",
            job_status_enum,
            nullable=False,
            server_default="pending",
            comment="Current job state",
        ),
        sa.Column(
            "progress",
            sa.Integer(),
            nullable=False,
            server_default="0",
            comment="Percentage complete (0-100)",
        ),
        sa.Column(
            "total_items",
            sa.Integer(),
            nullable=True,
            comment="Total items to process",
        ),
        sa.Column(
            "processed_items",
            sa.Integer(),
            nullable=False,
            server_default="0",
            comment="Items processed so far",
        ),
        sa.Column(
            "error", sa.Text(), nullable=True, comment="Error message if failed"
        ),
        sa.Column(
            "error_context",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
            comment="Diagnostic details: {stage, document_id, chunk_id, stack_trace}",
        ),
        sa.Column(
            "extra_data",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
            comment="Job parameters and configuration",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
            comment="When job was created",
        ),
        sa.Column(
            "started_at",
            sa.DateTime(timezone=True),
            nullable=True,
            comment="When processing began",
        ),
        sa.Column(
            "completed_at",
            sa.DateTime(timezone=True),
            nullable=True,
            comment="When processing finished (success or failure)",
        ),
        sa.CheckConstraint(
            "progress >= 0 AND progress <= 100", name="ck_jobs_progress_range"
        ),
        sa.CheckConstraint(
            "(total_items IS NULL) OR (processed_items <= total_items)",
            name="ck_jobs_processed_not_exceeds_total",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_jobs"),
    )
    op.create_index("ix_jobs_status", "jobs", ["status"], unique=False)
    op.create_index("ix_jobs_job_type", "jobs", ["job_type"], unique=False)
    op.create_index("ix_jobs_type_status", "jobs", ["job_type", "status"], unique=False)
    op.create_index(
        "ix_jobs_created_at", "jobs", [sa.text("created_at DESC")], unique=False
    )

    # --------------------------------------------------------------------------
    # audit_logs table
    # --------------------------------------------------------------------------
    op.create_table(
        "audit_logs",
        sa.Column("id", sa.UUID(), nullable=False, comment="UUID7 primary key"),
        sa.Column(
            "table_name",
            sa.String(length=100),
            nullable=False,
            comment="Name of the affected table",
        ),
        sa.Column(
            "record_id",
            sa.UUID(),
            nullable=False,
            comment="UUID of the affected record",
        ),
        sa.Column(
            "action",
            sa.String(length=20),
            nullable=False,
            comment="Type of action (DELETE, UPDATE, BULK_DELETE, MERGE)",
        ),
        sa.Column(
            "old_data",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
            comment="Complete record state before the action",
        ),
        sa.Column(
            "new_data",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
            comment="Record state after action (for updates)",
        ),
        sa.Column(
            "reason",
            sa.Text(),
            nullable=True,
            comment="Optional context for why this action occurred",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
            comment="When the action was logged",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_audit_logs"),
    )
    op.create_index(
        "ix_audit_logs_table_name", "audit_logs", ["table_name"], unique=False
    )
    op.create_index(
        "ix_audit_logs_record_id", "audit_logs", ["record_id"], unique=False
    )
    op.create_index("ix_audit_logs_action", "audit_logs", ["action"], unique=False)
    op.create_index(
        "ix_audit_logs_created_at",
        "audit_logs",
        [sa.text("created_at DESC")],
        unique=False,
    )
    op.create_index(
        "ix_audit_logs_table_record",
        "audit_logs",
        ["table_name", "record_id"],
        unique=False,
    )


def downgrade() -> None:
    """Drop all tables and enums in reverse order."""

    # Drop tables in reverse dependency order
    op.drop_table("audit_logs")
    op.drop_table("jobs")
    op.drop_table("evidence")
    op.drop_table("relations")
    op.drop_table("entities")
    op.drop_table("chunks")
    op.drop_table("documents")

    # Drop enum types
    op.execute("DROP TYPE IF EXISTS jobstatus")
    op.execute("DROP TYPE IF EXISTS relationlabel")
    op.execute("DROP TYPE IF EXISTS entitytype")