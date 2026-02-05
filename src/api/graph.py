"""Graph API endpoints for subgraph and path queries."""

from collections import defaultdict
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.core.logging import get_logger
from src.db import get_db
from src.db.models import Entity, Evidence, Relation
from src.schemas import (
    EntitySummary,
    EvidenceResponse,
    GraphEdge,
    GraphEdgeWithEvidence,
    GraphNode,
    PathResponse,
    PathsResponse,
    PathStep,
    RelationWithEntities,
    SubgraphResponse,
    SubgraphStats,
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


def entity_to_summary(entity: Entity) -> EntitySummary:
    """Convert entity to summary."""
    return EntitySummary(
        id=entity.id,
        name=entity.name,
        type=entity.type.value if hasattr(entity.type, "value") else str(entity.type),
        relation_count=0,
    )


def entity_to_node(entity: Entity, depth: int = 0) -> GraphNode:
    """Convert entity to graph node."""
    return GraphNode(
        id=entity.id,
        name=entity.name,
        type=entity.type.value if hasattr(entity.type, "value") else str(entity.type),
        depth=depth,
    )


# =============================================================================
# Endpoints
# =============================================================================


@router.get(
    "/subgraph",
    response_model=SubgraphResponse,
    summary="Get bounded subgraph",
    description="Retrieve a subgraph starting from an entity up to a specified depth.",
)
async def get_subgraph(
    entity_id: UUID = Query(description="Starting entity UUID"),
    depth: int = Query(default=1, ge=1, le=3, description="Max traversal depth"),
    max_nodes: int = Query(default=50, ge=1, le=200, description="Max nodes"),
    max_edges: int = Query(default=100, ge=1, le=500, description="Max edges"),
    relation_types: str | None = Query(
        default=None, description="Comma-separated relation types to filter"
    ),
    entity_types: str | None = Query(
        default=None, description="Comma-separated entity types to filter"
    ),
    min_confidence: float = Query(default=0.0, ge=0.0, le=1.0, description="Min confidence"),
    include_evidence: bool = Query(default=False, description="Include evidence"),
    db: AsyncSession = Depends(get_db),
) -> SubgraphResponse:
    """Get bounded subgraph from an entity."""
    # Validate starting entity exists
    start_entity = await get_entity_or_404(db, entity_id)

    # Parse filter lists
    relation_type_list = (
        [rt.strip() for rt in relation_types.split(",")]
        if relation_types
        else None
    )
    entity_type_list = (
        [et.strip() for et in entity_types.split(",")]
        if entity_types
        else None
    )

    # BFS traversal to collect subgraph
    visited_nodes: dict[UUID, tuple[Entity, int]] = {entity_id: (start_entity, 0)}
    collected_edges: list[Relation] = []
    frontier = {entity_id}

    for current_depth in range(depth):
        if not frontier:
            break

        next_frontier = set()

        # Get all relations connected to frontier
        for node_id in frontier:
            # Get outgoing relations
            outgoing_query = select(Relation).options(
                selectinload(Relation.tail_entity)
            ).where(
                Relation.head_id == node_id,
                Relation.confidence >= min_confidence,
            )

            # Get incoming relations
            incoming_query = select(Relation).options(
                selectinload(Relation.head_entity)
            ).where(
                Relation.tail_id == node_id,
                Relation.confidence >= min_confidence,
            )

            # Apply relation type filter
            if relation_type_list:
                from src.db.enums import RelationLabel

                valid_labels = []
                for rt in relation_type_list:
                    try:
                        valid_labels.append(RelationLabel(rt))
                    except ValueError:
                        pass
                if valid_labels:
                    outgoing_query = outgoing_query.where(
                        Relation.label.in_(valid_labels)
                    )
                    incoming_query = incoming_query.where(
                        Relation.label.in_(valid_labels)
                    )

            # Execute queries
            outgoing_result = await db.execute(outgoing_query)
            incoming_result = await db.execute(incoming_query)

            for rel in outgoing_result.scalars().all():
                neighbor = rel.tail_entity
                neighbor_type = (
                    neighbor.type.value
                    if hasattr(neighbor.type, "value")
                    else str(neighbor.type)
                )

                # Apply entity type filter
                if entity_type_list and neighbor_type not in entity_type_list:
                    continue

                if rel not in collected_edges and len(collected_edges) < max_edges:
                    collected_edges.append(rel)

                if neighbor.id not in visited_nodes and len(visited_nodes) < max_nodes:
                    visited_nodes[neighbor.id] = (neighbor, current_depth + 1)
                    next_frontier.add(neighbor.id)

            for rel in incoming_result.scalars().all():
                neighbor = rel.head_entity
                neighbor_type = (
                    neighbor.type.value
                    if hasattr(neighbor.type, "value")
                    else str(neighbor.type)
                )

                # Apply entity type filter
                if entity_type_list and neighbor_type not in entity_type_list:
                    continue

                if rel not in collected_edges and len(collected_edges) < max_edges:
                    collected_edges.append(rel)

                if neighbor.id not in visited_nodes and len(visited_nodes) < max_nodes:
                    visited_nodes[neighbor.id] = (neighbor, current_depth + 1)
                    next_frontier.add(neighbor.id)

        frontier = next_frontier

        # Check limits
        if len(visited_nodes) >= max_nodes or len(collected_edges) >= max_edges:
            break

    # Get evidence counts (or full evidence if requested)
    evidence_data: dict[UUID, list[Evidence]] = defaultdict(list)
    if collected_edges:
        edge_ids = [e.id for e in collected_edges]
        evidence_query = select(Evidence).where(Evidence.relation_id.in_(edge_ids))

        if include_evidence:
            evidence_result = await db.execute(evidence_query)
            for ev in evidence_result.scalars().all():
                evidence_data[ev.relation_id].append(ev)
        else:
            # Just get counts
            count_result = await db.execute(
                select(Evidence.relation_id, func.count(Evidence.id).label("count"))
                .where(Evidence.relation_id.in_(edge_ids))
                .group_by(Evidence.relation_id)
            )
            evidence_counts = {row.relation_id: row.count for row in count_result.all()}

    # Build response
    nodes = [
        entity_to_node(entity, node_depth)
        for entity, node_depth in visited_nodes.values()
    ]

    edges = []
    for rel in collected_edges:
        label = rel.label.value if hasattr(rel.label, "value") else str(rel.label)

        if include_evidence:
            edges.append(
                GraphEdgeWithEvidence(
                    id=rel.id,
                    source=rel.head_id,
                    target=rel.tail_id,
                    label=label,
                    confidence=rel.confidence,
                    evidence_count=len(evidence_data.get(rel.id, [])),
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
                        for ev in evidence_data.get(rel.id, [])
                    ],
                )
            )
        else:
            edges.append(
                GraphEdge(
                    id=rel.id,
                    source=rel.head_id,
                    target=rel.tail_id,
                    label=label,
                    confidence=rel.confidence,
                    evidence_count=evidence_counts.get(rel.id, 0)
                    if not include_evidence
                    else len(evidence_data.get(rel.id, [])),
                )
            )

    # Calculate stats
    nodes_by_type: dict[str, int] = defaultdict(int)
    for node in nodes:
        nodes_by_type[node.type] += 1

    edges_by_label: dict[str, int] = defaultdict(int)
    for edge in edges:
        edges_by_label[edge.label] += 1

    max_depth_reached = max((d for _, d in visited_nodes.values()), default=0)

    return SubgraphResponse(
        nodes=nodes,
        edges=edges,
        center_node=entity_id,
        depth=max_depth_reached,
        truncated=len(visited_nodes) >= max_nodes or len(collected_edges) >= max_edges,
        stats=SubgraphStats(
            node_count=len(nodes),
            edge_count=len(edges),
            nodes_by_type=dict(nodes_by_type),
            edges_by_label=dict(edges_by_label),
        ),
    )


@router.get(
    "/path",
    response_model=PathsResponse,
    summary="Find paths between entities",
    description="Find shortest paths between two entities using BFS.",
)
async def find_paths(
    source_id: UUID = Query(description="Source entity UUID"),
    target_id: UUID = Query(description="Target entity UUID"),
    max_hops: int = Query(default=3, ge=1, le=5, description="Max path length"),
    relation_types: str | None = Query(
        default=None, description="Comma-separated relation types to filter"
    ),
    min_confidence: float = Query(default=0.0, ge=0.0, le=1.0, description="Min confidence"),
    include_evidence: bool = Query(default=False, description="Include evidence"),
    db: AsyncSession = Depends(get_db),
) -> PathsResponse:
    """Find paths between two entities using BFS."""
    # Validate entities exist
    source_entity = await get_entity_or_404(db, source_id)
    target_entity = await get_entity_or_404(db, target_id)

    if source_id == target_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Source and target must be different entities",
        )

    # Parse relation types
    relation_type_list = (
        [rt.strip() for rt in relation_types.split(",")]
        if relation_types
        else None
    )

    # BFS to find paths
    # Each queue item: (current_node_id, path_so_far)
    # path_so_far is list of (entity_id, relation, direction)
    from collections import deque

    queue = deque([(source_id, [(source_id, None, None)])])
    visited = {source_id}
    found_paths: list[list[tuple]] = []

    while queue and len(found_paths) < 5:  # Limit to 5 paths
        current_id, path = queue.popleft()

        if len(path) > max_hops + 1:
            continue

        # Get connected relations
        outgoing_query = select(Relation).options(
            selectinload(Relation.tail_entity),
            selectinload(Relation.head_entity),
        ).where(
            Relation.head_id == current_id,
            Relation.confidence >= min_confidence,
        )

        incoming_query = select(Relation).options(
            selectinload(Relation.tail_entity),
            selectinload(Relation.head_entity),
        ).where(
            Relation.tail_id == current_id,
            Relation.confidence >= min_confidence,
        )

        # Apply relation type filter
        if relation_type_list:
            from src.db.enums import RelationLabel

            valid_labels = []
            for rt in relation_type_list:
                try:
                    valid_labels.append(RelationLabel(rt))
                except ValueError:
                    pass
            if valid_labels:
                outgoing_query = outgoing_query.where(Relation.label.in_(valid_labels))
                incoming_query = incoming_query.where(Relation.label.in_(valid_labels))

        outgoing_result = await db.execute(outgoing_query)
        incoming_result = await db.execute(incoming_query)

        # Process outgoing
        for rel in outgoing_result.scalars().all():
            neighbor_id = rel.tail_id

            if neighbor_id == target_id:
                # Found a path!
                complete_path = path + [(neighbor_id, rel, "forward")]
                found_paths.append(complete_path)
                continue

            if neighbor_id not in visited and len(path) < max_hops:
                visited.add(neighbor_id)
                queue.append((neighbor_id, path + [(neighbor_id, rel, "forward")]))

        # Process incoming
        for rel in incoming_result.scalars().all():
            neighbor_id = rel.head_id

            if neighbor_id == target_id:
                # Found a path!
                complete_path = path + [(neighbor_id, rel, "backward")]
                found_paths.append(complete_path)
                continue

            if neighbor_id not in visited and len(path) < max_hops:
                visited.add(neighbor_id)
                queue.append((neighbor_id, path + [(neighbor_id, rel, "backward")]))

    # Build response
    paths_response = []

    for path_data in found_paths:
        # Get all entity IDs in path
        entity_ids = [step[0] for step in path_data]
        entities_result = await db.execute(
            select(Entity).where(Entity.id.in_(entity_ids))
        )
        entities_map = {e.id: e for e in entities_result.scalars().all()}

        # Build steps
        steps = []
        total_confidence = 1.0

        for i, (entity_id, relation, direction) in enumerate(path_data):
            entity = entities_map.get(entity_id)
            if not entity:
                continue

            # Get evidence if requested and there's a relation
            evidence = []
            if include_evidence and relation:
                ev_result = await db.execute(
                    select(Evidence).where(Evidence.relation_id == relation.id)
                )
                evidence = [
                    EvidenceResponse(
                        id=ev.id,
                        relation_id=ev.relation_id,
                        chunk_id=ev.chunk_id,
                        quote=ev.quote,
                        quote_start=ev.quote_start,
                        quote_end=ev.quote_end,
                        created_at=ev.created_at,
                    )
                    for ev in ev_result.scalars().all()
                ]

            # Update confidence
            if relation:
                total_confidence *= relation.confidence

            # Build relation response for this step
            relation_response = None
            if relation and i < len(path_data) - 1:
                # Get next entity for relation context
                head_entity = entities_map.get(relation.head_id)
                tail_entity = entities_map.get(relation.tail_id)

                if head_entity and tail_entity:
                    relation_response = RelationWithEntities(
                        id=relation.id,
                        head_id=relation.head_id,
                        tail_id=relation.tail_id,
                        label=relation.label.value
                        if hasattr(relation.label, "value")
                        else str(relation.label),
                        confidence=relation.confidence,
                        created_at=relation.created_at,
                        head=entity_to_summary(head_entity),
                        tail=entity_to_summary(tail_entity),
                    )

            steps.append(
                PathStep(
                    entity=entity_to_summary(entity),
                    relation=relation_response,
                    direction=direction,
                    evidence=evidence,
                )
            )

        if steps:
            paths_response.append(
                PathResponse(
                    source=entity_to_summary(source_entity),
                    target=entity_to_summary(target_entity),
                    path_length=len(steps) - 1,
                    steps=steps,
                    total_confidence=total_confidence,
                )
            )

    shortest_length = min((p.path_length for p in paths_response), default=None)

    return PathsResponse(
        paths=paths_response,
        source=entity_to_summary(source_entity),
        target=entity_to_summary(target_entity),
        path_count=len(paths_response),
        shortest_path_length=shortest_length,
        searched_depth=max_hops,
    )


@router.get(
    "/stats",
    summary="Get graph statistics",
    description="Get overall graph statistics.",
)
async def get_graph_stats(
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Get overall graph statistics."""
    # Entity count
    entity_count = await db.execute(select(func.count(Entity.id)))

    # Relation count
    relation_count = await db.execute(select(func.count(Relation.id)))

    # Evidence count
    evidence_count = await db.execute(select(func.count(Evidence.id)))

    # Entities by type
    entity_types = await db.execute(
        select(Entity.type, func.count(Entity.id).label("count"))
        .group_by(Entity.type)
        .order_by(func.count(Entity.id).desc())
    )

    # Relations by label
    relation_labels = await db.execute(
        select(Relation.label, func.count(Relation.id).label("count"))
        .group_by(Relation.label)
        .order_by(func.count(Relation.id).desc())
    )

    # Average confidence
    avg_confidence = await db.execute(select(func.avg(Relation.confidence)))

    return {
        "entity_count": entity_count.scalar() or 0,
        "relation_count": relation_count.scalar() or 0,
        "evidence_count": evidence_count.scalar() or 0,
        "entities_by_type": {
            row.type.value if hasattr(row.type, "value") else str(row.type): row.count
            for row in entity_types.all()
        },
        "relations_by_label": {
            row.label.value if hasattr(row.label, "value") else str(row.label): row.count
            for row in relation_labels.all()
        },
        "average_confidence": round(avg_confidence.scalar() or 0, 3),
    }
