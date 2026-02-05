"""Entity API endpoints for searching and exploring entities."""

from collections import defaultdict
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import Text, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.core.logging import get_logger
from src.db import get_db
from src.db.models import Entity, Evidence, Relation
from src.schemas import (
    EntityNeighborhood,
    EntityResponse,
    EntitySummary,
    EntityTypeCount,
    EvidenceResponse,
    NeighborRelation,
    PaginatedResponse,
    RelationWithEvidence,
)

logger = get_logger(__name__)

router = APIRouter()


# =============================================================================
# Helper Functions
# =============================================================================


async def get_entity_or_404(db: AsyncSession, entity_id: UUID) -> Entity:
    """Get entity by ID or raise 404."""
    result = await db.execute(select(Entity).where(Entity.id == entity_id))
    entity = result.scalar_one_or_none()
    if not entity:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Entity {entity_id} not found",
        )
    return entity


def entity_to_summary(entity: Entity, relation_count: int = 0) -> EntitySummary:
    """Convert entity to summary."""
    return EntitySummary(
        id=entity.id,
        name=entity.name,
        type=entity.type.value if hasattr(entity.type, "value") else str(entity.type),
        relation_count=relation_count,
    )


# =============================================================================
# Endpoints
# =============================================================================


@router.get(
    "/search",
    response_model=PaginatedResponse[EntitySummary],
    summary="Search entities",
    description="Search entities by name or alias with optional type filter.",
)
async def search_entities(
    query: str = Query(
        min_length=1,
        max_length=200,
        description="Search query (matches name or aliases)",
    ),
    entity_type: str | None = Query(default=None, description="Filter by entity type"),
    page: int = Query(default=1, ge=1, description="Page number"),
    page_size: int = Query(default=20, ge=1, le=100, description="Items per page"),
    db: AsyncSession = Depends(get_db),
) -> PaginatedResponse[EntitySummary]:
    """Search entities by name or alias."""
    offset = (page - 1) * page_size
    search_pattern = f"%{query.lower()}%"

    # Build base query
    base_query = select(Entity).where(
        or_(
            Entity.normalized_name.ilike(search_pattern),
            Entity.name.ilike(search_pattern),
            # Search in aliases JSONB array
            Entity.aliases.cast(Text).ilike(search_pattern),
        )
    )

    # Add type filter
    if entity_type:
        from src.db.enums import EntityType

        try:
            type_enum = EntityType(entity_type)
            base_query = base_query.where(Entity.type == type_enum)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid entity type: {entity_type}",
            )

    # Get total count
    count_query = select(func.count()).select_from(base_query.subquery())
    count_result = await db.execute(count_query)
    total = count_result.scalar() or 0

    # Get entities with relation counts
    # Subquery for outgoing relations
    outgoing_count = (
        select(Relation.head_id, func.count(Relation.id).label("out_count"))
        .group_by(Relation.head_id)
        .subquery()
    )

    # Subquery for incoming relations
    incoming_count = (
        select(Relation.tail_id, func.count(Relation.id).label("in_count"))
        .group_by(Relation.tail_id)
        .subquery()
    )

    # Main query with counts
    query_stmt = (
        select(
            Entity,
            (
                func.coalesce(outgoing_count.c.out_count, 0)
                + func.coalesce(incoming_count.c.in_count, 0)
            ).label("relation_count"),
        )
        .outerjoin(outgoing_count, Entity.id == outgoing_count.c.head_id)
        .outerjoin(incoming_count, Entity.id == incoming_count.c.tail_id)
        .where(
            or_(
                Entity.normalized_name.ilike(search_pattern),
                Entity.name.ilike(search_pattern),
                Entity.aliases.cast(Text).ilike(search_pattern),
            )
        )
    )

    if entity_type:
        from src.db.enums import EntityType

        type_enum = EntityType(entity_type)
        query_stmt = query_stmt.where(Entity.type == type_enum)

    query_stmt = query_stmt.order_by(Entity.name).offset(offset).limit(page_size)

    result = await db.execute(query_stmt)
    rows = result.all()

    items = [
        EntitySummary(
            id=row.Entity.id,
            name=row.Entity.name,
            type=row.Entity.type.value
            if hasattr(row.Entity.type, "value")
            else str(row.Entity.type),
            relation_count=row.relation_count or 0,
        )
        for row in rows
    ]

    return PaginatedResponse.create(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get(
    "/types",
    response_model=list[EntityTypeCount],
    summary="Get entity type counts",
    description="Get count of entities grouped by type.",
)
async def get_entity_types(
    db: AsyncSession = Depends(get_db),
) -> list[EntityTypeCount]:
    """Get entity counts by type."""
    result = await db.execute(
        select(Entity.type, func.count(Entity.id).label("count"))
        .group_by(Entity.type)
        .order_by(func.count(Entity.id).desc())
    )

    return [
        EntityTypeCount(
            type=row.type.value if hasattr(row.type, "value") else str(row.type),
            count=row.count,
        )
        for row in result.all()
    ]


@router.get(
    "/{entity_id}",
    response_model=EntityResponse,
    summary="Get entity by ID",
    description="Retrieve a single entity with its details.",
)
async def get_entity(
    entity_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> EntityResponse:
    """Get entity by ID."""
    entity = await get_entity_or_404(db, entity_id)

    return EntityResponse(
        id=entity.id,
        name=entity.name,
        normalized_name=entity.normalized_name,
        type=entity.type.value if hasattr(entity.type, "value") else str(entity.type),
        description=entity.description,
        aliases=entity.aliases or [],
        external_ids=entity.external_ids or {},
        extra_data=entity.extra_data or {},
        created_at=entity.created_at,
        updated_at=entity.updated_at,
    )


@router.get(
    "/{entity_id}/neighborhood",
    response_model=EntityNeighborhood,
    summary="Get entity neighborhood",
    description="Get entity with all connected nodes and relations.",
)
async def get_entity_neighborhood(
    entity_id: UUID,
    limit: int = Query(default=50, ge=1, le=200, description="Max relations to return"),
    db: AsyncSession = Depends(get_db),
) -> EntityNeighborhood:
    """Get entity with its neighborhood (connected entities and relations)."""
    entity = await get_entity_or_404(db, entity_id)

    # Get outgoing relations (entity is head)
    outgoing_result = await db.execute(
        select(Relation)
        .options(selectinload(Relation.tail_entity))
        .where(Relation.head_id == entity_id)
        .limit(limit // 2)
    )
    outgoing_relations = outgoing_result.scalars().all()

    # Get incoming relations (entity is tail)
    incoming_result = await db.execute(
        select(Relation)
        .options(selectinload(Relation.head_entity))
        .where(Relation.tail_id == entity_id)
        .limit(limit // 2)
    )
    incoming_relations = incoming_result.scalars().all()

    # Get evidence counts for all relations
    all_relation_ids = [r.id for r in outgoing_relations] + [
        r.id for r in incoming_relations
    ]

    evidence_counts = {}
    if all_relation_ids:
        evidence_result = await db.execute(
            select(Evidence.relation_id, func.count(Evidence.id).label("count"))
            .where(Evidence.relation_id.in_(all_relation_ids))
            .group_by(Evidence.relation_id)
        )
        evidence_counts = {row.relation_id: row.count for row in evidence_result.all()}

    # Build neighbor relations
    neighbor_relations = []

    for rel in outgoing_relations:
        neighbor_relations.append(
            NeighborRelation(
                relation_id=rel.id,
                direction="outgoing",
                label=rel.label.value if hasattr(rel.label, "value") else str(rel.label),
                confidence=rel.confidence,
                neighbor=entity_to_summary(rel.tail_entity),
                evidence_count=evidence_counts.get(rel.id, 0),
            )
        )

    for rel in incoming_relations:
        neighbor_relations.append(
            NeighborRelation(
                relation_id=rel.id,
                direction="incoming",
                label=rel.label.value if hasattr(rel.label, "value") else str(rel.label),
                confidence=rel.confidence,
                neighbor=entity_to_summary(rel.head_entity),
                evidence_count=evidence_counts.get(rel.id, 0),
            )
        )

    # Calculate statistics
    relation_counts_by_label: dict[str, int] = defaultdict(int)
    neighbor_counts_by_type: dict[str, int] = defaultdict(int)

    for nr in neighbor_relations:
        relation_counts_by_label[nr.label] += 1
        neighbor_counts_by_type[nr.neighbor.type] += 1

    # Get total relation count
    total_result = await db.execute(
        select(func.count(Relation.id)).where(
            or_(Relation.head_id == entity_id, Relation.tail_id == entity_id)
        )
    )
    total_relations = total_result.scalar() or 0

    return EntityNeighborhood(
        entity=EntityResponse(
            id=entity.id,
            name=entity.name,
            normalized_name=entity.normalized_name,
            type=entity.type.value
            if hasattr(entity.type, "value")
            else str(entity.type),
            description=entity.description,
            aliases=entity.aliases or [],
            external_ids=entity.external_ids or {},
            extra_data=entity.extra_data or {},
            created_at=entity.created_at,
            updated_at=entity.updated_at,
        ),
        relations=neighbor_relations,
        total_relations=total_relations,
        relation_counts_by_label=dict(relation_counts_by_label),
        neighbor_counts_by_type=dict(neighbor_counts_by_type),
    )


@router.get(
    "/{entity_id}/relations",
    response_model=PaginatedResponse[RelationWithEvidence],
    summary="Get entity relations with evidence",
    description="Get all relations for an entity with supporting evidence.",
)
async def get_entity_relations(
    entity_id: UUID,
    direction: str = Query(
        default="both",
        description="Relation direction: 'outgoing', 'incoming', or 'both'",
    ),
    page: int = Query(default=1, ge=1, description="Page number"),
    page_size: int = Query(default=20, ge=1, le=100, description="Items per page"),
    db: AsyncSession = Depends(get_db),
) -> PaginatedResponse[RelationWithEvidence]:
    """Get relations for an entity with evidence."""
    await get_entity_or_404(db, entity_id)

    offset = (page - 1) * page_size

    # Build query based on direction
    if direction == "outgoing":
        condition = Relation.head_id == entity_id
    elif direction == "incoming":
        condition = Relation.tail_id == entity_id
    else:
        condition = or_(Relation.head_id == entity_id, Relation.tail_id == entity_id)

    # Get total count
    count_result = await db.execute(select(func.count(Relation.id)).where(condition))
    total = count_result.scalar() or 0

    # Get relations with entities and evidence
    result = await db.execute(
        select(Relation)
        .options(
            selectinload(Relation.head_entity),
            selectinload(Relation.tail_entity),
            selectinload(Relation.evidence_records),
        )
        .where(condition)
        .order_by(Relation.confidence.desc())
        .offset(offset)
        .limit(page_size)
    )
    relations = result.scalars().all()

    items = []
    for rel in relations:
        items.append(
            RelationWithEvidence(
                id=rel.id,
                head_id=rel.head_id,
                tail_id=rel.tail_id,
                label=rel.label.value if hasattr(rel.label, "value") else str(rel.label),
                confidence=rel.confidence,
                created_at=rel.created_at,
                head=entity_to_summary(rel.head_entity),
                tail=entity_to_summary(rel.tail_entity),
                evidence=[
                    EvidenceResponse(
                        id=ev.id,
                        relation_id=ev.relation_id,
                        chunk_id=ev.chunk_id,
                        quote=ev.quote,
                        quote_start=ev.quote_start,
                        quote_end=ev.quote_end,
                        created_at=ev.created_at,
                    )
                    for ev in rel.evidence_records
                ],
            )
        )

    return PaginatedResponse.create(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
    )
