"""
Chunk model for storing text segments prepared for LLM extraction.

Documents are split into chunks to:
1. Fit within LLM context windows
2. Enable granular evidence tracking (map quotes back to source)
3. Allow re-extraction without re-ingestion

Chunks preserve character offsets to reconstruct original positions.
"""

import uuid
from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey, Index, Integer, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.db.base import Base, CreatedAtMixin, UUIDMixin

if TYPE_CHECKING:
    from src.db.models.document import Document
    from src.db.models.evidence import Evidence


class Chunk(UUIDMixin, CreatedAtMixin, Base):
    """
    A text segment from a document, prepared for LLM extraction.

    Attributes:
        id: UUID7 primary key
        document_id: Foreign key to parent document
        chunk_index: Position in document (0, 1, 2, ...)
        text: The chunk content
        start_offset: Character start position in original document
        end_offset: Character end position in original document
        token_count: Approximate token count (for LLM context management)
        extra_data: Chunking parameters used (size, overlap, etc.)
        created_at: When the chunk was created

    Relationships:
        document: Parent document
        evidence_records: Evidence quotes from this chunk

    Constraints:
        - (document_id, chunk_index) must be unique

    Example:
        chunk = Chunk(
            document_id=doc.id,
            chunk_index=0,
            text="Ambroxol is a mucolytic agent that...",
            start_offset=0,
            end_offset=1000,
            token_count=250,
        )
    """

    # === Foreign Key ===
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Parent document",
    )

    # === Core Fields ===
    chunk_index: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="Position in document sequence (0-indexed)",
    )

    text: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="Chunk content",
    )

    start_offset: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="Character start position in original document",
    )

    end_offset: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="Character end position in original document",
    )

    # === Optional Fields ===
    token_count: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="Approximate token count for LLM context management",
    )

    extra_data: Mapped[dict | None] = mapped_column(
        JSONB,
        nullable=True,
        default=dict,
        comment="Chunking parameters (size, overlap, method)",
    )

    # === Relationships ===
    document: Mapped["Document"] = relationship(
        "Document",
        back_populates="chunks",
    )

    evidence_records: Mapped[list["Evidence"]] = relationship(
        "Evidence",
        back_populates="chunk",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    # === Table Constraints ===
    __table_args__ = (
        UniqueConstraint(
            "document_id",
            "chunk_index",
            name="uq_chunks_document_id_chunk_index",
        ),
    )

    def __repr__(self) -> str:
        return f"<Chunk(document_id={self.document_id}, index={self.chunk_index}, len={len(self.text)})>"

    @property
    def char_length(self) -> int:
        """Length of the chunk in characters."""
        return len(self.text)

    def contains_offset(self, offset: int) -> bool:
        """Check if a character offset falls within this chunk."""
        return self.start_offset <= offset < self.end_offset

    def get_relative_offset(self, absolute_offset: int) -> int | None:
        """
        Convert an absolute document offset to a chunk-relative offset.

        Returns None if the offset is not within this chunk.
        """
        if not self.contains_offset(absolute_offset):
            return None
        return absolute_offset - self.start_offset


# === Indexes ===
# Composite index for efficient document chunk retrieval in order
Index(
    "ix_chunks_document_id_chunk_index",
    Chunk.document_id,
    Chunk.chunk_index,
)
