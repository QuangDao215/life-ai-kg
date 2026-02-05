"""
Entity model for Knowledge Graph nodes.

Entities represent biomedical concepts extracted from text:
- Diseases (e.g., Parkinson's disease)
- Genes (e.g., GBA1)
- Proteins (e.g., GCase, alpha-synuclein)
- Compounds/Drugs (e.g., Ambroxol)
- And more...

Key features:
- Normalized names for deduplication and matching
- ENUM type for controlled vocabulary
- JSONB aliases for flexible name variations
- Unique constraint on (normalized_name, type) for idempotent upserts
"""

from typing import TYPE_CHECKING

from sqlalchemy import Enum, Index, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.db.base import Base, TimestampMixin, UUIDMixin
from src.db.enums import EntityType

if TYPE_CHECKING:
    from src.db.models.relation import Relation


class Entity(UUIDMixin, TimestampMixin, Base):
    """
    A node in the Knowledge Graph representing a biomedical concept.

    Attributes:
        id: UUID7 primary key
        name: Display name (original form from extraction)
        normalized_name: Canonical form for matching (lowercase, trimmed)
        type: Entity type from controlled vocabulary (EntityType enum)
        description: Optional description of the entity
        aliases: Alternative names/synonyms for this entity
        external_ids: External database identifiers (UMLS, MeSH, UniProt, etc.)
        extra_data: Additional attributes
        created_at: When first extracted
        updated_at: Last modification time

    Relationships:
        outgoing_relations: Relations where this entity is the head (source)
        incoming_relations: Relations where this entity is the tail (target)

    Constraints:
        - (normalized_name, type) must be unique

    Normalization:
        The normalized_name is computed by:
        1. Lowercase conversion
        2. Whitespace normalization (collapse multiple spaces)
        3. Leading/trailing whitespace removal
        4. Punctuation trimming

    Example:
        entity = Entity(
            name="Parkinson's Disease",
            normalized_name="parkinson's disease",
            type=EntityType.DISEASE,
            aliases=["PD", "Parkinson disease", "paralysis agitans"],
        )
    """

    # === Core Fields ===
    name: Mapped[str] = mapped_column(
        String(500),
        nullable=False,
        comment="Display name (original form)",
    )

    normalized_name: Mapped[str] = mapped_column(
        String(500),
        nullable=False,
        index=True,
        comment="Canonical form for matching (lowercased, trimmed)",
    )

    type: Mapped[EntityType] = mapped_column(
        Enum(
            EntityType,
            name="entitytype",
            create_constraint=False,
            native_enum=True,
            values_callable=lambda e: [member.value for member in e],
        ),
        nullable=False,
        index=True,
        comment="Entity type from controlled vocabulary",
    )

    # === Optional Fields ===
    description: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="Description of the entity",
    )

    aliases: Mapped[list | None] = mapped_column(
        JSONB,
        nullable=True,
        default=list,
        comment="Alternative names/synonyms",
    )

    external_ids: Mapped[dict | None] = mapped_column(
        JSONB,
        nullable=True,
        default=dict,
        comment="External identifiers {source: id} e.g., {umls: C0030567}",
    )

    extra_data: Mapped[dict | None] = mapped_column(
        JSONB,
        nullable=True,
        default=dict,
        comment="Additional attributes",
    )

    # === Relationships ===
    # Relations where this entity is the source (head)
    outgoing_relations: Mapped[list["Relation"]] = relationship(
        "Relation",
        back_populates="head_entity",
        foreign_keys="Relation.head_id",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    # Relations where this entity is the target (tail)
    incoming_relations: Mapped[list["Relation"]] = relationship(
        "Relation",
        back_populates="tail_entity",
        foreign_keys="Relation.tail_id",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    # === Table Constraints ===
    __table_args__ = (
        # Unique constraint for idempotent upserts
        # Same normalized name + type = same entity
        Index(
            "uq_entities_normalized_name_type",
            "normalized_name",
            "type",
            unique=True,
        ),
    )

    def __repr__(self) -> str:
        return f"<Entity(name={self.name!r}, type={self.type.value})>"

    @property
    def all_names(self) -> list[str]:
        """Get all names including primary name and aliases."""
        names = [self.name]
        if self.aliases:
            names.extend(self.aliases)
        return names

    @property
    def relation_count(self) -> int:
        """Total number of relations involving this entity."""
        return len(self.outgoing_relations) + len(self.incoming_relations)

    def matches_name(self, query: str) -> bool:
        """
        Check if query matches this entity's name or aliases.

        Case-insensitive matching against:
        - Primary name
        - Normalized name
        - All aliases
        """
        query_lower = query.lower().strip()
        if query_lower in self.normalized_name:
            return True
        if query_lower in self.name.lower():
            return True
        if self.aliases:
            for alias in self.aliases:
                if query_lower in alias.lower():
                    return True
        return False

    def add_alias(self, alias: str) -> bool:
        """
        Add an alias if it doesn't already exist.

        Returns True if alias was added, False if already present.
        """
        if self.aliases is None:
            self.aliases = []

        # Check if alias already exists (case-insensitive)
        alias_lower = alias.lower().strip()
        existing_lower = [a.lower() for a in self.aliases]

        if alias_lower not in existing_lower and alias_lower != self.normalized_name:
            self.aliases.append(alias)
            return True
        return False


# === Indexes ===
# Index for searching by name (case-insensitive via application)
Index("ix_entities_name", Entity.name)

# Index for type-based filtering
Index("ix_entities_type", Entity.type)
