"""
Controlled vocabulary enums for the Knowledge Graph.

This module defines the allowed types for:
- Entity types (nodes in the KG)
- Relation labels (edges in the KG)
- Job statuses (async pipeline states)

These enums enforce schema compliance at the database level.
Any extraction that produces types/labels outside this vocabulary
will be rejected (REQ-SC-03).

Reference:
- REQ-SC-01: Entity types
- REQ-SC-02: Relation labels
- REQ-KG-06: Job status tracking
"""

from enum import Enum


class EntityType(str, Enum):
    """
    Controlled vocabulary for entity types in the Knowledge Graph.

    These represent the categories of biomedical concepts that can be
    extracted from text and stored as nodes in the KG.

    From requirements (REQ-SC-01):
    - Disease
    - Gene
    - Protein
    - Pathway
    - CellType
    - Tissue/Region
    - Compound/Drug
    - Phenotype
    - Biomarker

    Usage:
        entity = Entity(
            name="Ambroxol",
            type=EntityType.COMPOUND_DRUG
        )

    The string values match the extraction schema for direct mapping
    from LLM outputs.
    """

    DISEASE = "Disease"
    GENE = "Gene"
    PROTEIN = "Protein"
    PATHWAY = "Pathway"
    CELL_TYPE = "CellType"
    TISSUE_REGION = "Tissue/Region"
    COMPOUND_DRUG = "Compound/Drug"
    PHENOTYPE = "Phenotype"
    BIOMARKER = "Biomarker"

    @classmethod
    def from_string(cls, value: str) -> "EntityType | None":
        """
        Convert a string to EntityType, with fuzzy matching.

        Handles common variations in LLM outputs:
        - Case insensitivity: "disease" -> DISEASE
        - Underscore/space: "Cell Type" -> CELL_TYPE
        - Slash variations: "Tissue/Region" or "Tissue_Region"

        Returns None if no match found (caller should handle rejection).

        Examples:
            EntityType.from_string("Disease")        -> EntityType.DISEASE
            EntityType.from_string("compound/drug")  -> EntityType.COMPOUND_DRUG
            EntityType.from_string("cell type")      -> EntityType.CELL_TYPE
            EntityType.from_string("invalid")        -> None
        """
        if not value:
            return None

        # Normalize: lowercase, replace spaces/underscores with consistent format
        normalized = value.strip().lower()

        # Direct match attempt
        for member in cls:
            if member.value.lower() == normalized:
                return member

        # Fuzzy matching for common variations
        normalized_alt = normalized.replace(" ", "").replace("_", "").replace("/", "")
        for member in cls:
            member_alt = member.value.lower().replace(" ", "").replace("_", "").replace("/", "")
            if member_alt == normalized_alt:
                return member

        return None

    @classmethod
    def values(cls) -> list[str]:
        """Return list of all valid entity type values."""
        return [member.value for member in cls]


class RelationLabel(str, Enum):
    """
    Controlled vocabulary for relation labels in the Knowledge Graph.

    These represent the types of relationships that can exist between
    entities. Each relation is a directed edge from head -> tail.

    From requirements (REQ-SC-02):
    - ASSOCIATED_WITH: General association
    - INCREASES_RISK: Head increases risk of tail
    - DECREASES_RISK: Head decreases risk of tail
    - UPREGULATES: Head increases expression/activity of tail
    - DOWNREGULATES: Head decreases expression/activity of tail
    - ACTIVATES: Head activates tail
    - INHIBITS: Head inhibits tail
    - BINDS: Head physically binds to tail
    - BIOMARKER_FOR: Head is a biomarker for tail
    - PART_OF: Head is part of tail
    - EXPRESSED_IN: Head is expressed in tail (gene/protein in tissue/cell)

    Usage:
        relation = Relation(
            head_id=ambroxol_id,
            tail_id=gcase_id,
            label=RelationLabel.ACTIVATES
        )

    Semantics:
        The relation is always read as "HEAD [LABEL] TAIL"
        Example: "Ambroxol ACTIVATES GCase"
    """

    ASSOCIATED_WITH = "ASSOCIATED_WITH"
    INCREASES_RISK = "INCREASES_RISK"
    DECREASES_RISK = "DECREASES_RISK"
    UPREGULATES = "UPREGULATES"
    DOWNREGULATES = "DOWNREGULATES"
    ACTIVATES = "ACTIVATES"
    INHIBITS = "INHIBITS"
    BINDS = "BINDS"
    BIOMARKER_FOR = "BIOMARKER_FOR"
    PART_OF = "PART_OF"
    EXPRESSED_IN = "EXPRESSED_IN"

    @classmethod
    def from_string(cls, value: str) -> "RelationLabel | None":
        """
        Convert a string to RelationLabel, with normalization.

        Handles common variations:
        - Case insensitivity: "activates" -> ACTIVATES
        - Spaces vs underscores: "associated with" -> ASSOCIATED_WITH

        Returns None if no match found.

        Examples:
            RelationLabel.from_string("ACTIVATES")        -> RelationLabel.ACTIVATES
            RelationLabel.from_string("associated_with")  -> RelationLabel.ASSOCIATED_WITH
            RelationLabel.from_string("binds to")         -> None (not exact match)
        """
        if not value:
            return None

        # Normalize: uppercase, replace spaces with underscores
        normalized = value.strip().upper().replace(" ", "_").replace("-", "_")

        # Direct match
        for member in cls:
            if member.value == normalized:
                return member

        return None

    @classmethod
    def values(cls) -> list[str]:
        """Return list of all valid relation label values."""
        return [member.value for member in cls]

    @property
    def is_directional(self) -> bool:
        """
        Check if this relation type is inherently directional.

        Directional relations have different meaning if reversed:
        - "A INHIBITS B" ≠ "B INHIBITS A"

        Symmetric relations are equivalent when reversed:
        - "A ASSOCIATED_WITH B" ≈ "B ASSOCIATED_WITH A"
        - "A BINDS B" ≈ "B BINDS A"

        This is useful for graph queries where direction matters.
        """
        symmetric = {
            RelationLabel.ASSOCIATED_WITH,
            RelationLabel.BINDS,
        }
        return self not in symmetric


class JobStatus(str, Enum):
    """
    Status values for async pipeline jobs.

    Job lifecycle::

        PENDING -> RUNNING -> COMPLETED
                          |-> FAILED
                          |-> CANCELLED

    From requirements (REQ-KG-06):
    - Track status, progress, and error details
    - Support checking status/progress/failure reasons

    Usage:
        job = Job(
            job_type="kg_build",
            status=JobStatus.PENDING
        )

        # Update during processing
        job.status = JobStatus.RUNNING
        job.progress = 50

        # On completion
        job.status = JobStatus.COMPLETED
        job.progress = 100
    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        """
        Check if this status represents a terminal state.

        Terminal states mean the job will not change status again:
        - COMPLETED: Successfully finished
        - FAILED: Ended with error
        - CANCELLED: Manually stopped

        Non-terminal states may still transition:
        - PENDING: Waiting to start
        - RUNNING: Currently processing
        """
        return self in {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED}

    @property
    def is_success(self) -> bool:
        """Check if this status represents successful completion."""
        return self == JobStatus.COMPLETED

    @property
    def is_error(self) -> bool:
        """Check if this status represents an error state."""
        return self == JobStatus.FAILED

    @classmethod
    def values(cls) -> list[str]:
        """Return list of all valid job status values."""
        return [member.value for member in cls]


# =============================================================================
# VALIDATION HELPERS
# =============================================================================

def is_valid_entity_type(value: str) -> bool:
    """Check if a string is a valid entity type."""
    return EntityType.from_string(value) is not None


def is_valid_relation_label(value: str) -> bool:
    """Check if a string is a valid relation label."""
    return RelationLabel.from_string(value) is not None


def validate_entity_type(value: str) -> EntityType:
    """
    Validate and convert a string to EntityType.

    Raises ValueError if the value is not a valid entity type.
    Use this when you want to fail fast on invalid input.

    Example:
        try:
            entity_type = validate_entity_type(llm_output["type"])
        except ValueError as e:
            logger.warning(f"Invalid entity type: {e}")
            # Skip this entity
    """
    result = EntityType.from_string(value)
    if result is None:
        valid_types = ", ".join(EntityType.values())
        raise ValueError(f"Invalid entity type: '{value}'. Must be one of: {valid_types}")
    return result


def validate_relation_label(value: str) -> RelationLabel:
    """
    Validate and convert a string to RelationLabel.

    Raises ValueError if the value is not a valid relation label.

    Example:
        try:
            label = validate_relation_label(llm_output["relation"])
        except ValueError as e:
            logger.warning(f"Invalid relation: {e}")
            # Skip this relation
    """
    result = RelationLabel.from_string(value)
    if result is None:
        valid_labels = ", ".join(RelationLabel.values())
        raise ValueError(f"Invalid relation label: '{value}'. Must be one of: {valid_labels}")
    return result
