"""
Document model for storing PubMed publications.

Documents are the source material from which entities and relations
are extracted. Each document corresponds to a PubMed article with
its metadata (title, abstract, authors, etc.).

Key features:
- PMID as unique identifier (idempotent ingestion)
- JSONB for flexible metadata storage
- Relationship to chunks for text segmentation
"""

from datetime import date
from typing import TYPE_CHECKING

from sqlalchemy import Date, Index, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.db.base import Base, CreatedAtMixin, UUIDMixin

if TYPE_CHECKING:
    from src.db.models.chunk import Chunk


class Document(UUIDMixin, CreatedAtMixin, Base):
    """
    A source document (PubMed publication) in the knowledge graph.

    Attributes:
        id: UUID7 primary key
        pmid: PubMed ID (unique identifier for idempotent ingestion)
        title: Article title
        abstract: Article abstract (main text for extraction)
        authors: List of author objects [{"name": "...", "affiliation": "..."}]
        journal: Journal name
        publication_date: Date of publication
        doi: Digital Object Identifier (optional)
        mesh_terms: MeSH terms assigned to the article
        extra_data: Additional flexible metadata (keywords, etc.)
        created_at: When the document was ingested

    Relationships:
        chunks: Text segments created from this document

    Example:
        document = Document(
            pmid="12345678",
            title="Ambroxol and GCase in Parkinson's Disease",
            abstract="Background: Ambroxol is a mucolytic agent...",
            authors=[{"name": "Smith J", "affiliation": "MIT"}],
            journal="Neurology",
            publication_date=date(2023, 5, 15),
        )
    """

    # === Core Fields ===
    pmid: Mapped[str] = mapped_column(
        String(20),
        unique=True,
        nullable=False,
        index=True,
        comment="PubMed ID - unique identifier for idempotent ingestion",
    )

    title: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="Article title",
    )

    abstract: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="Article abstract - main source for entity extraction",
    )

    # === Metadata Fields ===
    authors: Mapped[list | None] = mapped_column(
        JSONB,
        nullable=True,
        default=list,
        comment="List of authors [{name, affiliation}]",
    )

    journal: Mapped[str | None] = mapped_column(
        String(500),
        nullable=True,
        comment="Journal name",
    )

    publication_date: Mapped[date | None] = mapped_column(
        Date,
        nullable=True,
        comment="Publication date",
    )

    doi: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
        comment="Digital Object Identifier",
    )

    mesh_terms: Mapped[list | None] = mapped_column(
        JSONB,
        nullable=True,
        default=list,
        comment="MeSH terms assigned to this article",
    )

    extra_data: Mapped[dict | None] = mapped_column(
        JSONB,
        nullable=True,
        default=dict,
        comment="Additional metadata (keywords, affiliations, etc.)",
    )

    # === Relationships ===
    chunks: Mapped[list["Chunk"]] = relationship(
        "Chunk",
        back_populates="document",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    def __repr__(self) -> str:
        return f"<Document(pmid={self.pmid!r}, title={self.title[:50]!r}...)>"

    @property
    def full_text(self) -> str:
        """Combine title and abstract for extraction."""
        parts = [self.title]
        if self.abstract:
            parts.append(self.abstract)
        return "\n\n".join(parts)

    @property
    def author_names(self) -> list[str]:
        """Extract just the author names from the authors list."""
        if not self.authors:
            return []
        return [a.get("name", "") for a in self.authors if a.get("name")]


# === Indexes ===
# Index on publication_date for filtering by date range
Index("ix_documents_publication_date", Document.publication_date)
