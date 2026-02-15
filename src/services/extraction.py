"""
Entity and relation extraction service.

This module provides LLM-based extraction of biomedical entities and
relationships from text chunks, with evidence linking for provenance.

Features:
- Entity extraction with type classification
- Relation extraction with confidence scores
- Evidence linking with exact quote offsets
- Batch processing with progress tracking
- Idempotent database storage

Requirements addressed:
- REQ-KG-01: Extract entities from text
- REQ-KG-02: Extract relations between entities
- REQ-SC-04: Evidence quotes with offsets
- REQ-KG-05: Idempotent upserts
"""

import asyncio
import json
import re
from dataclasses import dataclass, field
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.config import settings
from src.core.logging import get_logger
from src.db.enums import EntityType, RelationLabel, is_valid_entity_type, is_valid_relation_label
from src.db.models import Chunk, Document, Entity, Evidence, Relation
from src.services.llm_client import BaseLLMClient, LLMMessage, LLMParseError, get_llm_client

logger = get_logger(__name__)


# =============================================================================
# Prompts
# =============================================================================

ENTITY_EXTRACTION_SYSTEM_PROMPT = """You are a biomedical named entity recognition expert. Extract all biomedical entities from the given text.

For each entity, provide:
1. name: The exact text mention in the source
2. type: One of the following types ONLY:
   - Disease: Medical conditions, disorders, syndromes (e.g., "Parkinson's disease", "Gaucher disease")
   - Gene: Gene names and symbols (e.g., "GBA1", "SNCA", "LRRK2")
   - Protein: Protein names and enzymes (e.g., "glucocerebrosidase", "α-synuclein", "GCase")
   - Pathway: Biological pathways (e.g., "autophagy", "lysosomal pathway")
   - CellType: Cell types (e.g., "dopaminergic neurons", "microglia")
   - Tissue/Region: Anatomical regions (e.g., "substantia nigra", "brain")
   - Compound/Drug: Chemical compounds and drugs (e.g., "Ambroxol", "levodopa")
   - Phenotype: Observable characteristics (e.g., "motor symptoms", "tremor")
   - Biomarker: Biological markers (e.g., "CSF α-synuclein", "GCase activity")
3. normalized_name: A standardized lowercase identifier (e.g., "parkinsons_disease", "alpha_synuclein")

Rules:
- Extract ALL relevant biomedical entities, even if mentioned multiple times
- Use the EXACT text from the source for the name field
- Be specific with entity types - don't confuse genes with proteins
- GBA1/GBA is a Gene, glucocerebrosidase/GCase is a Protein
- α-synuclein/alpha-synuclein is a Protein, SNCA is its Gene

Respond with ONLY a valid JSON object in this exact format:
{
  "entities": [
    {"name": "exact text", "type": "EntityType", "normalized_name": "standardized_id"},
    ...
  ]
}"""

RELATION_EXTRACTION_SYSTEM_PROMPT = """You are a biomedical relation extraction expert. Given a text and a list of entities, extract relationships between them.

For each relation, provide:
1. head: The source entity name (exact text from entities list)
2. tail: The target entity name (exact text from entities list)
3. label: One of the following relation types ONLY:
   - ASSOCIATED_WITH: General association between entities
   - INCREASES_RISK: Head increases risk of tail (e.g., mutation increases disease risk)
   - DECREASES_RISK: Head decreases risk of tail
   - UPREGULATES: Head increases expression/activity of tail
   - DOWNREGULATES: Head decreases expression/activity of tail
   - ACTIVATES: Head activates tail
   - INHIBITS: Head inhibits tail
   - BINDS: Head physically binds to tail
   - BIOMARKER_FOR: Head is a biomarker for tail
   - PART_OF: Head is part of tail
   - EXPRESSED_IN: Head is expressed in tail (gene/protein in tissue/cell)
4. confidence: Your confidence score from 0.0 to 1.0
5. evidence: A word-for-word quote copied directly from the text. Must be a complete,
   contiguous passage — do NOT use "..." or ellipsis to skip content.

Rules:
- Only extract relations that are EXPLICITLY stated or strongly implied in the text
- The evidence field must be an EXACT, VERBATIM copy-paste from the source text
- NEVER use "..." or ellipsis to shorten or skip parts of the quote
- NEVER paraphrase, reword, reorder, or add/remove any words
- The quote must be a single contiguous passage that appears in the text exactly as written
- If the supporting text is too long, select a shorter but complete sentence instead
- Head and tail must be from the provided entities list
- Be conservative with confidence - use 0.9+ only for very explicit statements
- Prefer specific relations over ASSOCIATED_WITH when appropriate

Respond with ONLY a valid JSON object in this exact format:
{
  "relations": [
    {
      "head": "entity name",
      "tail": "entity name",
      "label": "RELATION_LABEL",
      "confidence": 0.85,
      "evidence": "exact quote from text"
    },
    ...
  ]
}"""


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class ExtractedEntity:
    """An entity extracted from text."""

    name: str
    type: str
    normalized_name: str

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "type": self.type,
            "normalized_name": self.normalized_name,
        }


@dataclass
class ExtractedRelation:
    """A relation extracted from text."""

    head: str
    tail: str
    label: str
    confidence: float
    evidence: str

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "head": self.head,
            "tail": self.tail,
            "label": self.label,
            "confidence": self.confidence,
            "evidence": self.evidence,
        }


@dataclass
class ExtractionResult:
    chunk_id: UUID
    document_id: UUID
    entities: list[ExtractedEntity] = field(default_factory=list)
    relations: list[ExtractedRelation] = field(default_factory=list)
    error: str | None = None

    @property
    def entity_count(self) -> int:
        return len(self.entities)

    @property
    def relation_count(self) -> int:
        return len(self.relations)


@dataclass
class BatchExtractionResult:
    """Result of batch extraction."""

    total_chunks: int = 0
    processed_chunks: int = 0
    total_entities: int = 0
    total_relations: int = 0
    new_entities: int = 0
    new_relations: int = 0
    errors: int = 0
    results: list[ExtractionResult] = field(default_factory=list)


# =============================================================================
# Extraction Service
# =============================================================================


class ExtractionService:
    """
    Service for extracting entities and relations from document chunks.

    Handles:
    - LLM-based entity extraction
    - LLM-based relation extraction
    - Evidence linking with quote offsets
    - Database storage with idempotent upserts

    Usage:
        async with get_llm_client() as llm:
            service = ExtractionService(db, llm)
            result = await service.extract_from_chunk(chunk)
    """

    def __init__(
        self,
        db: AsyncSession,
        llm_client: BaseLLMClient,
        confidence_threshold: float | None = None,
        max_entities_per_chunk: int | None = None,
        max_relations_per_chunk: int | None = None,
        chunk_delay_seconds: int = 2,
        call_delay_seconds: int = 1,
    ):
        """
        Initialize extraction service.

        Args:
            db: Database session
            llm_client: LLM client for extraction
            confidence_threshold: Minimum confidence for relations
            max_entities_per_chunk: Maximum entities to extract
            max_relations_per_chunk: Maximum relations to extract
            chunk_delay_seconds: Delay between processing chunks (rate limit)
            call_delay_seconds: Delay between API calls within a chunk
        """
        self.db = db
        self.llm = llm_client
        self.confidence_threshold = (
            confidence_threshold or settings.extraction_confidence_threshold
        )
        self.max_entities = max_entities_per_chunk or settings.max_entities_per_chunk
        self.max_relations = max_relations_per_chunk or settings.max_relations_per_chunk
        self.chunk_delay = chunk_delay_seconds
        self.call_delay = call_delay_seconds

        # Cache for entity lookups
        self._entity_cache: dict[tuple[str, str], Entity] = {}

    # =========================================================================
    # Entity Extraction
    # =========================================================================

    async def extract_entities(self, text: str) -> list[ExtractedEntity]:
        """
        Extract entities from text using LLM.

        Args:
            text: Input text

        Returns:
            List of extracted entities
        """
        messages = [
            LLMMessage(role="system", content=ENTITY_EXTRACTION_SYSTEM_PROMPT),
            LLMMessage(
                role="user",
                content=f"Extract all biomedical entities from this text:\n\n{text}",
            ),
        ]

        response = await self.llm.complete(messages, temperature=0.0)

        # Parse response
        entities = self._parse_entity_response(response.content)

        # Limit entities if needed
        if len(entities) > self.max_entities:
            logger.warning(
                "Truncating entities",
                extracted=len(entities),
                max=self.max_entities,
            )
            entities = entities[: self.max_entities]

        return entities

    def _parse_entity_response(self, response: str) -> list[ExtractedEntity]:
        """Parse LLM response for entities."""
        try:
            # Extract JSON from response (handle markdown code blocks)
            json_str = self._extract_json(response)
            json_str = self._repair_json(json_str)
            data = json.loads(json_str)

            entities = []
            for item in data.get("entities", []):
                # Validate entity type
                entity_type = item.get("type", "")
                if not is_valid_entity_type(entity_type):
                    logger.warning("Invalid entity type", type=entity_type)
                    continue

                entities.append(
                    ExtractedEntity(
                        name=item.get("name", ""),
                        type=entity_type,
                        normalized_name=item.get("normalized_name", "").lower(),
                    )
                )

            return entities

        except (json.JSONDecodeError, KeyError) as e:
            logger.error("Failed to parse entity response", error=str(e))
            raise LLMParseError(f"Failed to parse entity response: {e}") from e

    # =========================================================================
    # Relation Extraction
    # =========================================================================

    async def extract_relations(
        self,
        text: str,
        entities: list[ExtractedEntity],
    ) -> list[ExtractedRelation]:
        """
        Extract relations between entities using LLM.

        Args:
            text: Input text
            entities: List of entities to find relations between

        Returns:
            List of extracted relations
        """
        if len(entities) < 2:
            return []

        # Format entities for prompt
        entity_list = "\n".join(
            f"- {e.name} ({e.type})" for e in entities
        )

        messages = [
            LLMMessage(role="system", content=RELATION_EXTRACTION_SYSTEM_PROMPT),
            LLMMessage(
                role="user",
                content=f"""Extract relations between these entities from the text.

ENTITIES:
{entity_list}

TEXT:
{text}""",
            ),
        ]

        response = await self.llm.complete(messages, temperature=0.0)

        # Parse response
        relations = self._parse_relation_response(response.content)

        # Filter by confidence threshold
        relations = [
            r for r in relations if r.confidence >= self.confidence_threshold
        ]

        # Limit relations if needed
        if len(relations) > self.max_relations:
            logger.warning(
                "Truncating relations",
                extracted=len(relations),
                max=self.max_relations,
            )
            # Keep highest confidence relations
            relations = sorted(relations, key=lambda r: r.confidence, reverse=True)
            relations = relations[: self.max_relations]

        return relations

    def _parse_relation_response(self, response: str) -> list[ExtractedRelation]:
        """Parse LLM response for relations."""
        try:
            json_str = self._extract_json(response)
            json_str = self._repair_json(json_str)
            data = json.loads(json_str)

            relations = []
            for item in data.get("relations", []):
                # Validate relation label
                label = item.get("label", "")
                if not is_valid_relation_label(label):
                    logger.warning("Invalid relation label", label=label)
                    continue

                relations.append(
                    ExtractedRelation(
                        head=item.get("head", ""),
                        tail=item.get("tail", ""),
                        label=label,
                        confidence=float(item.get("confidence", 0.5)),
                        evidence=item.get("evidence", ""),
                    )
                )

            return relations

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(
                "Failed to parse relation response",
                error=str(e),
                raw_response=response[:2000],  # First 500 chars
            )
            raise LLMParseError(f"Failed to parse relation response: {e}") from e

    def _extract_json(self, text: str) -> str:
        """Extract JSON from text, handling markdown code blocks."""
        # Try to find JSON in code blocks
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if json_match:
            return json_match.group(1)

        # Try to find raw JSON object
        json_match = re.search(r"(\{.*\})", text, re.DOTALL)
        if json_match:
            return json_match.group(1)

        return text

    def _repair_json(self, json_str: str) -> str:
        """
        Attempt to repair common JSON issues from LLM responses.

        Fixes:
        - Trailing commas before ] or }
        - Missing commas between objects
        - Single quotes instead of double quotes
        """
        # Remove trailing commas before ] or }
        json_str = re.sub(r",\s*([}\]])", r"\1", json_str)

        # Fix missing commas between objects (} followed by { without comma)
        json_str = re.sub(r"}\s*{", r"},{", json_str)

        # Fix missing commas after string values followed by "
        # This catches: "value" "key": -> "value", "key":
        json_str = re.sub(r'"\s*\n\s*"', '",\n"', json_str)

        # Fix missing commas after numbers followed by "
        json_str = re.sub(r'(\d)\s*\n\s*"', r'\1,\n"', json_str)

        # Fix missing commas after } followed by "
        json_str = re.sub(r'}\s*\n\s*"', r'},\n"', json_str)

        try:
            json.loads(json_str)
            return json_str
        except json.JSONDecodeError:
            # Find last complete object (last '},' or '}]')
            last_complete = json_str.rfind('},')
            if last_complete > 0:
                truncated = json_str[:last_complete + 1]  # Keep up to the last complete '}'
                truncated += ']}'  # Close the array and outer object
                return truncated

        return json_str

    # =========================================================================
    # Chunk Processing
    # =========================================================================

    async def extract_from_chunk(
        self,
        chunk: Chunk,
        save_to_db: bool = True,
    ) -> ExtractionResult:
        """
        Extract entities and relations from a single chunk.

        Args:
            chunk: Chunk to process
            save_to_db: Whether to save results to database

        Returns:
            ExtractionResult with extracted data
        """
        # Capture chunk attributes early to avoid greenlet issues after rollback
        chunk_id = chunk.id
        chunk_id_str = str(chunk_id)
        chunk_doc_id = chunk.document_id
        chunk_text = chunk.text

        result = ExtractionResult(
            chunk_id=chunk_id,
            document_id=chunk_doc_id,
        )

        try:
            # Extract entities
            entities = await self.extract_entities(chunk_text)
            result.entities = entities

            logger.debug(
                "Extracted entities",
                chunk_id=chunk_id_str,
                count=len(entities),
            )

            # Extract relations (only if we have entities)
            relations = []  # Initialize to avoid unbound variable
            if entities:
                # Add delay between entity and relation calls to avoid rate limiting
                await asyncio.sleep(self.call_delay)

                try:
                    relations = await self.extract_relations(chunk_text, entities)
                    result.relations = relations

                    logger.debug(
                        "Extracted relations",
                        chunk_id=chunk_id_str,
                        count=len(relations),
                    )
                except Exception as rel_error:
                    # Log relation error but continue with entities
                    logger.warning(
                        "Relation extraction failed, saving entities only",
                        chunk_id=chunk_id_str,
                        error=str(rel_error),
                    )
                    result.error = f"Relation extraction failed: {rel_error}"
                    # Relations stay empty, but we still save entities

            # Save to database (save entities even if relations failed)
            if save_to_db and entities:
                await self._save_extraction(chunk, entities, relations)

        except Exception as e:
            logger.error(
                "Extraction failed",
                chunk_id=chunk_id_str,
                error=str(e),
            )
            result.error = str(e)

        return result

    async def _extract_from_chunk_data(
        self,
        chunk_data: dict,
        save_to_db: bool = True,
    ) -> ExtractionResult:
        """
        Extract entities and relations from chunk data dict.

        This method works with pre-extracted chunk data (dict) instead of
        ORM objects, which avoids greenlet issues after rollback.

        Args:
            chunk_data: Dict with id, document_id, text keys
            save_to_db: Whether to save results to database

        Returns:
            ExtractionResult with extracted data
        """
        chunk_id = chunk_data["id"]
        chunk_id_str = str(chunk_id)
        chunk_doc_id = chunk_data["document_id"]
        chunk_text = chunk_data["text"]

        result = ExtractionResult(
            chunk_id=chunk_id,
            document_id=chunk_doc_id,
        )

        try:
            # Extract entities
            entities = await self.extract_entities(chunk_text)
            result.entities = entities

            logger.debug(
                "Extracted entities",
                chunk_id=chunk_id_str,
                count=len(entities),
            )

            # Extract relations (only if we have entities)
            relations = []  # Initialize to avoid unbound variable
            if entities:
                # Add delay between entity and relation calls to avoid rate limiting
                await asyncio.sleep(self.call_delay)

                try:
                    relations = await self.extract_relations(chunk_text, entities)
                    result.relations = relations

                    logger.debug(
                        "Extracted relations",
                        chunk_id=chunk_id_str,
                        count=len(relations),
                    )
                except Exception as rel_error:
                    # Log relation error but continue with entities
                    logger.warning(
                        "Relation extraction failed, saving entities only",
                        chunk_id=chunk_id_str,
                        error=str(rel_error),
                    )
                    result.error = f"Relation extraction failed: {rel_error}"
                    # Relations stay empty, but we still save entities

            # Save to database (save entities even if relations failed)
            if save_to_db and entities:
                await self._save_extraction_from_data(chunk_data, entities, relations)

        except Exception as e:
            logger.error(
                "Extraction failed",
                chunk_id=chunk_id_str,
                error=str(e),
            )
            result.error = str(e)

        return result

    async def _save_extraction_from_data(
        self,
        chunk_data: dict,
        entities: list[ExtractedEntity],
        relations: list[ExtractedRelation],
    ) -> None:
        """Save extracted entities and relations to database using chunk data dict."""
        chunk_id = chunk_data["id"]
        chunk_text = chunk_data["text"]

        # Create/get entities
        entity_map: dict[str, Entity] = {}

        for ext_entity in entities:
            entity = await self._get_or_create_entity(ext_entity)
            entity_map[ext_entity.name] = entity

        # Track evidence we've created in this batch to avoid duplicates
        # Key: (relation_id, quote_hash)
        created_evidence: set[tuple] = set()

        # Create relations with evidence
        for ext_relation in relations:
            head_entity = entity_map.get(ext_relation.head)
            tail_entity = entity_map.get(ext_relation.tail)

            if not head_entity or not tail_entity:
                logger.warning(
                    "Relation entity not found",
                    head=ext_relation.head,
                    tail=ext_relation.tail,
                )
                continue

            # Create or get relation
            relation = await self._get_or_create_relation(
                head_entity,
                tail_entity,
                ext_relation.label,
                ext_relation.confidence,
            )

            # Skip duplicate evidence within this batch
            quote_hash = Evidence.compute_hash(ext_relation.evidence)
            evidence_key = (relation.id, quote_hash)
            if evidence_key in created_evidence:
                logger.debug(
                    "Skipping duplicate evidence in batch",
                    relation_id=str(relation.id),
                    quote=ext_relation.evidence[:50],
                )
                continue
            created_evidence.add(evidence_key)

            # Create evidence linking relation to chunk
            await self._create_evidence_from_data(
                relation=relation,
                chunk_id=chunk_id,
                chunk_text=chunk_text,
                quote=ext_relation.evidence,
            )

    async def _save_extraction(
        self,
        chunk: Chunk,
        entities: list[ExtractedEntity],
        relations: list[ExtractedRelation],
    ) -> None:
        """Save extracted entities and relations to database."""
        # Create/get entities
        entity_map: dict[str, Entity] = {}

        for ext_entity in entities:
            entity = await self._get_or_create_entity(ext_entity)
            entity_map[ext_entity.name] = entity

        # Track evidence we've created in this batch to avoid duplicates
        # Key: (relation_id, quote_hash)
        created_evidence: set[tuple] = set()

        # Create relations with evidence
        for ext_relation in relations:
            head_entity = entity_map.get(ext_relation.head)
            tail_entity = entity_map.get(ext_relation.tail)

            if not head_entity or not tail_entity:
                logger.warning(
                    "Relation entity not found",
                    head=ext_relation.head,
                    tail=ext_relation.tail,
                )
                continue

            # Create or get relation
            relation = await self._get_or_create_relation(
                head_entity,
                tail_entity,
                ext_relation.label,
                ext_relation.confidence,
            )

            # Create evidence record
            if ext_relation.evidence:
                # Skip duplicate evidence within this batch
                quote_hash = Evidence.compute_hash(ext_relation.evidence)
                evidence_key = (relation.id, quote_hash)
                if evidence_key in created_evidence:
                    logger.debug(
                        "Skipping duplicate evidence in batch",
                        relation_id=str(relation.id),
                        quote=ext_relation.evidence[:50],
                    )
                    continue
                created_evidence.add(evidence_key)

                await self._create_evidence(
                    relation,
                    chunk,
                    ext_relation.evidence,
                )

        await self.db.flush()

    async def _get_or_create_entity(self, ext_entity: ExtractedEntity) -> Entity:
        """Get existing entity or create new one."""
        # Check cache first
        cache_key = (ext_entity.normalized_name, ext_entity.type)
        if cache_key in self._entity_cache:
            return self._entity_cache[cache_key]

        # Get entity type enum
        entity_type = EntityType.from_string(ext_entity.type)
        if entity_type is None:
            raise ValueError(f"Invalid entity type: {ext_entity.type}")

        # Check database
        result = await self.db.execute(
            select(Entity).where(
                Entity.normalized_name == ext_entity.normalized_name,
                Entity.type == entity_type,
            )
        )
        entity = result.scalar_one_or_none()

        if entity:
            # Update aliases if new name variant found
            if ext_entity.name not in (entity.name, *(entity.aliases or [])):
                entity.add_alias(ext_entity.name)
        else:
            # Create new entity
            entity = Entity(
                name=ext_entity.name,
                normalized_name=ext_entity.normalized_name,
                type=entity_type,
            )
            self.db.add(entity)
            await self.db.flush()

        # Cache entity
        self._entity_cache[cache_key] = entity
        return entity

    async def _get_or_create_relation(
        self,
        head: Entity,
        tail: Entity,
        label: str,
        confidence: float,
    ) -> Relation:
        """Get existing relation or create new one."""
        # Get relation label enum
        relation_label = RelationLabel.from_string(label)
        if relation_label is None:
            raise ValueError(f"Invalid relation label: {label}")

        # Check for existing relation
        result = await self.db.execute(
            select(Relation).where(
                Relation.head_id == head.id,
                Relation.tail_id == tail.id,
                Relation.label == relation_label,
            )
        )
        relation = result.scalar_one_or_none()

        if relation:
            # Update confidence if higher
            if confidence > relation.confidence:
                relation.confidence = confidence
        else:
            # Create new relation
            relation = Relation(
                head_id=head.id,
                tail_id=tail.id,
                label=relation_label,
                confidence=confidence,
            )
            self.db.add(relation)
            await self.db.flush()

        return relation

    async def _create_evidence(
        self,
        relation: Relation,
        chunk: Chunk,
        quote: str,
    ) -> Evidence | None:
        """Create evidence record with quote offsets."""
        # Find quote in chunk text
        quote_start = chunk.text.find(quote)
        if quote_start == -1:
            # Try case-insensitive search
            quote_lower = quote.lower()
            text_lower = chunk.text.lower()
            quote_start = text_lower.find(quote_lower)

            if quote_start == -1:
                logger.warning(
                    "Evidence quote not found in chunk",
                    quote=quote[:50],
                    chunk_id=str(chunk.id),
                )
                # Still create evidence without exact offsets
                quote_start = None
                quote_end = None
            else:
                quote_end = quote_start + len(quote)
        else:
            quote_end = quote_start + len(quote)

        # Check for existing evidence with same quote
        quote_hash = Evidence.compute_hash(quote)
        result = await self.db.execute(
            select(Evidence).where(
                Evidence.relation_id == relation.id,
                Evidence.quote_hash == quote_hash,
            )
        )
        existing = result.scalar_one_or_none()

        if existing:
            return existing

        # Create evidence using factory method
        evidence = Evidence.create(
            relation_id=relation.id,
            chunk_id=chunk.id,
            quote=quote,
            quote_start=quote_start,
            quote_end=quote_end,
        )
        self.db.add(evidence)

        return evidence

    async def _create_evidence_from_data(
        self,
        relation: Relation,
        chunk_id,
        chunk_text: str,
        quote: str,
    ) -> Evidence | None:
        """Create evidence record with quote offsets (using chunk data, not ORM object)."""
        chunk_id_str = str(chunk_id)

        # Find quote in chunk text
        quote_start = chunk_text.find(quote)
        if quote_start == -1:
            # Try case-insensitive search
            quote_lower = quote.lower()
            text_lower = chunk_text.lower()
            quote_start = text_lower.find(quote_lower)

            if quote_start == -1:
                logger.warning(
                    "Evidence quote not found in chunk",
                    quote=quote[:50],
                    chunk_id=chunk_id_str,
                )
                # Still create evidence without exact offsets
                quote_start = None
                quote_end = None
            else:
                quote_end = quote_start + len(quote)
        else:
            quote_end = quote_start + len(quote)

        # Check for existing evidence with same quote
        quote_hash = Evidence.compute_hash(quote)
        result = await self.db.execute(
            select(Evidence).where(
                Evidence.relation_id == relation.id,
                Evidence.quote_hash == quote_hash,
            )
        )
        existing = result.scalar_one_or_none()

        if existing:
            return existing

        # Create evidence using factory method
        evidence = Evidence.create(
            relation_id=relation.id,
            chunk_id=chunk_id,
            quote=quote,
            quote_start=quote_start,
            quote_end=quote_end,
        )
        self.db.add(evidence)

        return evidence

    # =========================================================================
    # Batch Processing
    # =========================================================================

    async def extract_from_document(
        self,
        document: Document,
        save_to_db: bool = True,
    ) -> list[ExtractionResult]:
        """
        Extract from all chunks of a document.

        Args:
            document: Document to process
            save_to_db: Whether to save results

        Returns:
            List of ExtractionResult for each chunk
        """
        # Get document chunks
        result = await self.db.execute(
            select(Chunk)
            .where(Chunk.document_id == document.id)
            .order_by(Chunk.chunk_index)
        )
        chunks = result.scalars().all()

        results = []
        for chunk in chunks:
            extraction = await self.extract_from_chunk(chunk, save_to_db=save_to_db)
            results.append(extraction)

        if save_to_db:
            await self.db.commit()

        return results

    async def extract_all(
        self,
        limit: int | None = None,
        skip_processed: bool = True,
        progress_callback: callable | None = None,
    ) -> BatchExtractionResult:
        """
        Extract from all chunks in the database.

        Args:
            limit: Maximum number of chunks to process
            skip_processed: Skip chunks that already have extractions
            progress_callback: Optional callback(processed, total)

        Returns:
            BatchExtractionResult with summary statistics
        """
        # Get chunks to process
        query = select(Chunk)

        if skip_processed:
            # Get chunks that don't have any evidence records yet
            subquery = select(Evidence.chunk_id).distinct()
            query = query.where(~Chunk.id.in_(subquery))

        if limit:
            query = query.limit(limit)

        result = await self.db.execute(query)
        chunks = result.scalars().all()

        # CRITICAL: Extract all chunk data BEFORE the loop starts
        # After any rollback, ALL ORM objects become expired, including
        # chunks we haven't processed yet. Accessing expired attributes
        # triggers a sync DB refresh which fails in async context (greenlet error).
        chunk_data_list = [
            {
                "id": chunk.id,
                "document_id": chunk.document_id,
                "text": chunk.text,
                "chunk_index": chunk.chunk_index,
            }
            for chunk in chunks
        ]

        batch_result = BatchExtractionResult(total_chunks=len(chunk_data_list))

        logger.info("Starting batch extraction", total_chunks=len(chunk_data_list))

        for i, chunk_data in enumerate(chunk_data_list):
            chunk_id_str = str(chunk_data["id"])

            try:
                extraction = await self._extract_from_chunk_data(chunk_data, save_to_db=True)
                batch_result.results.append(extraction)
                batch_result.processed_chunks += 1
                batch_result.total_entities += extraction.entity_count
                batch_result.total_relations += extraction.relation_count

                if extraction.error:
                    batch_result.errors += 1
                    # Rollback failed transaction and start fresh
                    await self.db.rollback()
                    # Clear entity cache - rolled back entities no longer exist!
                    self._entity_cache.clear()

                if progress_callback:
                    progress_callback(i + 1, len(chunk_data_list))

                # Commit after each successful chunk to avoid cascading failures
                if not extraction.error:
                    await self.db.commit()

                # Add delay between chunks to avoid rate limiting
                await asyncio.sleep(self.chunk_delay)

            except Exception as e:
                logger.error("Chunk processing failed", chunk_id=chunk_id_str, error=str(e))
                batch_result.errors += 1
                # Rollback and continue with next chunk
                await self.db.rollback()
                # Clear entity cache - rolled back entities no longer exist!
                self._entity_cache.clear()
                if progress_callback:
                    progress_callback(i + 1, len(chunk_data_list))

        # Count new entities and relations
        entity_count = await self.db.execute(select(Entity))
        batch_result.new_entities = len(entity_count.scalars().all())

        relation_count = await self.db.execute(select(Relation))
        batch_result.new_relations = len(relation_count.scalars().all())

        logger.info(
            "Batch extraction complete",
            processed=batch_result.processed_chunks,
            entities=batch_result.total_entities,
            relations=batch_result.total_relations,
            errors=batch_result.errors,
        )

        return batch_result


# =============================================================================
# Convenience Functions
# =============================================================================


async def extract_from_text(
    text: str,
    llm_client: BaseLLMClient | None = None,
) -> tuple[list[ExtractedEntity], list[ExtractedRelation]]:
    """
    Convenience function to extract entities and relations from text.

    Args:
        text: Input text
        llm_client: Optional LLM client (creates one if not provided)

    Returns:
        Tuple of (entities, relations)

    Example:
        entities, relations = await extract_from_text(
            "GBA1 mutations increase risk of Parkinson's disease"
        )
    """
    if llm_client is None:
        llm_client = get_llm_client()

    async with llm_client:
        # Create a minimal service without database
        service = ExtractionService(
            db=None,  # type: ignore
            llm_client=llm_client,
        )

        entities = await service.extract_entities(text)
        relations = await service.extract_relations(text, entities)

        return entities, relations
