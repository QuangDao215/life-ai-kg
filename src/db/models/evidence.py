"""
Evidence model for linking relations to source text.

Evidence records provide provenance for extracted relations:
- What text snippet supports this relation?
- Which chunk/document did it come from?
- Where exactly in the text (character offsets)?

Key features:
- Mandatory quote field (REQ-SC-04)
- Hash-based deduplication to prevent storing identical quotes
- Character offsets for UI highlighting
- One relation can have multiple evidence records

Requirements addressed:
- REQ-SC-04: Mandatory evidence quotes for all relations
- REQ-KG-05: Idempotent evidence storage via quote_hash
"""

import hashlib
import uuid
from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.db.base import Base, CreatedAtMixin, UUIDMixin

if TYPE_CHECKING:
    from src.db.models.chunk import Chunk
    from src.db.models.relation import Relation


class Evidence(UUIDMixin, CreatedAtMixin, Base):
    """
    A provenance record linking a relation to its source text.

    Attributes:
        id: UUID7 primary key
        relation_id: The relation this evidence supports
        chunk_id: The source chunk containing the evidence
        quote: The evidence text (required - REQ-SC-04)
        quote_start: Character offset within chunk (optional)
        quote_end: Character offset within chunk (optional)
        quote_hash: SHA256 hash of quote for deduplication
        created_at: When the evidence was extracted

    Relationships:
        relation: The supported relation
        chunk: The source chunk

    Constraints:
        - (relation_id, chunk_id, quote_hash) must be unique
        - quote is mandatory (NOT NULL)

    Example:
        evidence = Evidence(
            relation_id=relation.id,
            chunk_id=chunk.id,
            quote="Ambroxol has been shown to enhance GCase activity",
            quote_start=150,
            quote_end=201,
        )

    Note:
        The quote_hash is automatically computed from the quote
        to enable efficient deduplication without comparing full text.
    """

    # Explicit table name (evidence is uncountable, not "evidences")
    __tablename__ = "evidence"

    # === Foreign Keys ===
    relation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("relations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="The relation this evidence supports",
    )

    chunk_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("chunks.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="The source chunk containing this evidence",
    )

    # === Core Fields ===
    quote: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="Evidence text supporting the relation (REQUIRED)",
    )

    quote_start: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="Character start offset within chunk",
    )

    quote_end: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="Character end offset within chunk",
    )

    quote_hash: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        index=True,
        comment="SHA256 hash of quote for deduplication",
    )

    # === Relationships ===
    relation: Mapped["Relation"] = relationship(
        "Relation",
        back_populates="evidence_records",
    )

    chunk: Mapped["Chunk"] = relationship(
        "Chunk",
        back_populates="evidence_records",
    )

    # === Table Constraints ===
    __table_args__ = (
        # Unique constraint for idempotent evidence storage
        # Same relation + chunk + quote = same evidence
        UniqueConstraint(
            "relation_id",
            "chunk_id",
            "quote_hash",
            name="uq_evidence_relation_chunk_quote",
        ),
    )

    def __repr__(self) -> str:
        quote_preview = self.quote[:50] + "..." if len(self.quote) > 50 else self.quote
        return f"<Evidence(relation={self.relation_id}, quote={quote_preview!r})>"

    @staticmethod
    def compute_hash(quote: str) -> str:
        """
        Compute SHA256 hash of a quote for deduplication.

        The hash is computed on the normalized quote:
        - Stripped of leading/trailing whitespace
        - Consistent encoding (UTF-8)

        Args:
            quote: The evidence text

        Returns:
            64-character hex string (SHA256)
        """
        normalized = quote.strip()
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    @classmethod
    def create(
        cls,
        relation_id: uuid.UUID,
        chunk_id: uuid.UUID,
        quote: str,
        quote_start: int | None = None,
        quote_end: int | None = None,
    ) -> "Evidence":
        """
        Factory method to create Evidence with auto-computed hash.

        Use this instead of direct construction to ensure quote_hash
        is properly computed.

        Args:
            relation_id: ID of the relation this evidence supports
            chunk_id: ID of the source chunk
            quote: The evidence text
            quote_start: Optional character offset start
            quote_end: Optional character offset end

        Returns:
            New Evidence instance with computed quote_hash
        """
        return cls(
            relation_id=relation_id,
            chunk_id=chunk_id,
            quote=quote.strip(),
            quote_start=quote_start,
            quote_end=quote_end,
            quote_hash=cls.compute_hash(quote),
        )

    @property
    def has_offsets(self) -> bool:
        """Check if character offsets are available."""
        return self.quote_start is not None and self.quote_end is not None

    @property
    def quote_length(self) -> int:
        """Length of the quote in characters."""
        return len(self.quote)

    def validate_in_chunk(self, chunk_text: str) -> bool:
        """
        Validate that this quote exists in the given chunk text.

        Args:
            chunk_text: The text content of the source chunk

        Returns:
            True if quote is found in chunk_text
        """
        return self.quote in chunk_text


# === Indexes ===
# Index for finding all evidence for a relation
Index("ix_evidence_relation_id", Evidence.relation_id)

# Index for finding all evidence from a chunk
Index("ix_evidence_chunk_id", Evidence.chunk_id)

# Index for hash-based lookups
Index("ix_evidence_quote_hash", Evidence.quote_hash)
