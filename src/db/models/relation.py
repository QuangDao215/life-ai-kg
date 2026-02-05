"""
Relation model for Knowledge Graph edges.

Relations represent directed connections between entities:
- "Ambroxol ACTIVATES GCase"
- "GBA1 mutations INCREASES_RISK Parkinson's disease"
- "alpha-synuclein ASSOCIATED_WITH neurodegeneration"

Key features:
- Directed edges (head -> tail)
- ENUM labels for controlled vocabulary
- Confidence scores from extraction
- Unique constraint on (head_id, tail_id, label) prevents duplicates
"""

import uuid
from typing import TYPE_CHECKING

from sqlalchemy import CheckConstraint, Float, ForeignKey, Index
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.db.base import Base, CreatedAtMixin, UUIDMixin
from src.db.enums import RelationLabel

if TYPE_CHECKING:
    from src.db.models.entity import Entity
    from src.db.models.evidence import Evidence


class Relation(UUIDMixin, CreatedAtMixin, Base):
    """
    A directed edge in the Knowledge Graph connecting two entities.

    Attributes:
        id: UUID7 primary key
        head_id: Source entity (subject of the relation)
        tail_id: Target entity (object of the relation)
        label: Relation type from controlled vocabulary
        confidence: Extraction confidence score (0.0 to 1.0)
        extra_data: Additional attributes (extraction source, etc.)
        created_at: When first extracted

    Relationships:
        head_entity: Source entity
        tail_entity: Target entity
        evidence_records: Supporting evidence for this relation

    Constraints:
        - (head_id, tail_id, label) must be unique
        - confidence must be between 0.0 and 1.0

    Semantics:
        Relations are always read as: HEAD [LABEL] TAIL
        Example: "Ambroxol [ACTIVATES] GCase"

    Example:
        relation = Relation(
            head_id=ambroxol.id,
            tail_id=gcase.id,
            label=RelationLabel.ACTIVATES,
            confidence=0.85,
        )
    """

    # === Foreign Keys ===
    head_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("entities.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Source entity (subject)",
    )

    tail_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("entities.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Target entity (object)",
    )

    # === Core Fields ===
    label: Mapped[RelationLabel] = mapped_column(
        nullable=False,
        index=True,
        comment="Relation type from controlled vocabulary",
    )

    confidence: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment="Extraction confidence score (0.0 to 1.0)",
    )

    # === Optional Fields ===
    extra_data: Mapped[dict | None] = mapped_column(
        JSONB,
        nullable=True,
        default=dict,
        comment="Additional attributes (extraction method, etc.)",
    )

    # === Relationships ===
    head_entity: Mapped["Entity"] = relationship(
        "Entity",
        back_populates="outgoing_relations",
        foreign_keys=[head_id],
    )

    tail_entity: Mapped["Entity"] = relationship(
        "Entity",
        back_populates="incoming_relations",
        foreign_keys=[tail_id],
    )

    evidence_records: Mapped[list["Evidence"]] = relationship(
        "Evidence",
        back_populates="relation",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    # === Table Constraints ===
    __table_args__ = (
        # Unique constraint for idempotent upserts
        # Same head + tail + label = same relation
        Index(
            "uq_relations_head_tail_label",
            "head_id",
            "tail_id",
            "label",
            unique=True,
        ),
        # Confidence must be between 0 and 1 if provided
        CheckConstraint(
            "(confidence IS NULL) OR (confidence >= 0.0 AND confidence <= 1.0)",
            name="ck_relations_confidence_range",
        ),
    )

    def __repr__(self) -> str:
        return f"<Relation(head={self.head_id}, {self.label.value}, tail={self.tail_id})>"

    @property
    def evidence_count(self) -> int:
        """Number of evidence records supporting this relation."""
        return len(self.evidence_records)

    @property
    def has_evidence(self) -> bool:
        """Check if this relation has at least one evidence record."""
        return len(self.evidence_records) > 0

    @property
    def is_high_confidence(self, threshold: float = 0.7) -> bool:
        """Check if confidence exceeds threshold."""
        return self.confidence is not None and self.confidence >= threshold

    @property
    def triple(self) -> tuple[uuid.UUID, str, uuid.UUID]:
        """Return the relation as a (head_id, label, tail_id) triple."""
        return (self.head_id, self.label.value, self.tail_id)

    def format_triple(self, head_name: str | None = None, tail_name: str | None = None) -> str:
        """
        Format the relation as a human-readable triple.

        Args:
            head_name: Name of head entity (uses ID if not provided)
            tail_name: Name of tail entity (uses ID if not provided)

        Returns:
            Formatted string like "Ambroxol ACTIVATES GCase"
        """
        head = head_name or str(self.head_id)[:8]
        tail = tail_name or str(self.tail_id)[:8]
        return f"{head} {self.label.value} {tail}"


# === Indexes ===
# Index for graph traversal from head
Index("ix_relations_head_id", Relation.head_id)

# Index for graph traversal to tail
Index("ix_relations_tail_id", Relation.tail_id)

# Index for filtering by relation type
Index("ix_relations_label", Relation.label)

# Composite index for finding relations between specific entities
Index(
    "ix_relations_head_tail",
    Relation.head_id,
    Relation.tail_id,
)
