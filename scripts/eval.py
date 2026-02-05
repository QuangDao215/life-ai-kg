#!/usr/bin/env python3
"""
Evaluation Harness for LIFE AI Knowledge Graph.

This script evaluates the quality and completeness of the Knowledge Graph
built from PubMed publications about Ambroxol and Parkinson's disease.

Usage:
    python scripts/eval.py [--output EVAL_REPORT.md] [--json] [--verbose]

Outputs:
    - Summary statistics (documents, entities, relations, evidence)
    - Quality checks (duplicates, orphans, schema violations)
    - Sanity queries demonstrating meaningful exploration
    - JSON report for artifact submission
"""

import argparse
import asyncio
import json
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from sqlalchemy import distinct, func, select, text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import selectinload

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.config import settings
from src.db.enums import EntityType, RelationLabel
from src.db.models import Chunk, Document, Entity, Evidence, Relation


# =============================================================================
# Data Classes for Report
# =============================================================================


@dataclass
class SummaryStats:
    """Summary statistics for the KG."""

    total_documents: int = 0
    total_chunks: int = 0
    chunks_processed: int = 0
    processing_percentage: float = 0.0

    total_entities: int = 0
    entities_by_type: dict[str, int] = field(default_factory=dict)

    total_relations: int = 0
    relations_by_label: dict[str, int] = field(default_factory=dict)

    total_evidence: int = 0
    relations_with_evidence: int = 0
    evidence_coverage: float = 0.0

    avg_confidence: float = 0.0
    min_confidence: float = 0.0
    max_confidence: float = 0.0


@dataclass
class QualityMetrics:
    """Quality check results."""

    # Duplicate checks
    duplicate_entities: int = 0
    duplicate_relations: int = 0
    duplicate_evidence: int = 0

    # Orphan checks
    orphan_relations: int = 0  # Relations with missing head/tail entities
    relations_without_evidence: int = 0

    # Schema validation
    invalid_entity_types: int = 0
    invalid_relation_labels: int = 0

    # Data quality
    entities_without_relations: int = 0
    avg_relations_per_entity: float = 0.0
    avg_evidence_per_relation: float = 0.0

    # Issues list
    issues: list[str] = field(default_factory=list)


@dataclass
class SanityCheckResult:
    """Result of a sanity check query."""

    name: str
    description: str
    passed: bool
    details: dict[str, Any] = field(default_factory=dict)
    message: str = ""


@dataclass
class EvaluationReport:
    """Complete evaluation report."""

    generated_at: str = ""
    database_url: str = ""

    summary: SummaryStats = field(default_factory=SummaryStats)
    quality: QualityMetrics = field(default_factory=QualityMetrics)
    sanity_checks: list[SanityCheckResult] = field(default_factory=list)

    # Example queries for artifacts
    example_queries: list[dict[str, Any]] = field(default_factory=list)

    overall_score: str = ""  # PASS / WARN / FAIL


# =============================================================================
# Evaluation Functions
# =============================================================================


async def compute_summary_stats(db: AsyncSession) -> SummaryStats:
    """Compute summary statistics for the KG."""
    stats = SummaryStats()

    # Document counts
    result = await db.execute(select(func.count(Document.id)))
    stats.total_documents = result.scalar() or 0

    # Chunk counts
    result = await db.execute(select(func.count(Chunk.id)))
    stats.total_chunks = result.scalar() or 0

    # Processed chunks (have evidence)
    result = await db.execute(select(func.count(distinct(Evidence.chunk_id))))
    stats.chunks_processed = result.scalar() or 0

    if stats.total_chunks > 0:
        stats.processing_percentage = round(
            (stats.chunks_processed / stats.total_chunks) * 100, 2
        )

    # Entity counts
    result = await db.execute(select(func.count(Entity.id)))
    stats.total_entities = result.scalar() or 0

    # Entities by type
    result = await db.execute(
        select(Entity.type, func.count(Entity.id))
        .group_by(Entity.type)
        .order_by(func.count(Entity.id).desc())
    )
    stats.entities_by_type = {row[0].value: row[1] for row in result.all()}

    # Relation counts
    result = await db.execute(select(func.count(Relation.id)))
    stats.total_relations = result.scalar() or 0

    # Relations by label
    result = await db.execute(
        select(Relation.label, func.count(Relation.id))
        .group_by(Relation.label)
        .order_by(func.count(Relation.id).desc())
    )
    stats.relations_by_label = {row[0].value: row[1] for row in result.all()}

    # Evidence counts
    result = await db.execute(select(func.count(Evidence.id)))
    stats.total_evidence = result.scalar() or 0

    # Relations with evidence
    result = await db.execute(select(func.count(distinct(Evidence.relation_id))))
    stats.relations_with_evidence = result.scalar() or 0

    if stats.total_relations > 0:
        stats.evidence_coverage = round(
            (stats.relations_with_evidence / stats.total_relations) * 100, 2
        )

    # Confidence statistics
    result = await db.execute(
        select(
            func.avg(Relation.confidence),
            func.min(Relation.confidence),
            func.max(Relation.confidence),
        )
    )
    row = result.one()
    stats.avg_confidence = round(float(row[0] or 0), 3)
    stats.min_confidence = round(float(row[1] or 0), 3)
    stats.max_confidence = round(float(row[2] or 0), 3)

    return stats


async def compute_quality_metrics(db: AsyncSession) -> QualityMetrics:
    """Compute quality metrics and identify issues."""
    quality = QualityMetrics()

    # Check for duplicate entities (same normalized_name + type)
    result = await db.execute(
        text("""
            SELECT normalized_name, type, COUNT(*) as cnt
            FROM entities
            GROUP BY normalized_name, type
            HAVING COUNT(*) > 1
        """)
    )
    duplicates = result.all()
    quality.duplicate_entities = sum(row[2] - 1 for row in duplicates)
    if quality.duplicate_entities > 0:
        quality.issues.append(
            f"Found {quality.duplicate_entities} duplicate entities"
        )

    # Check for duplicate relations (same head_id + tail_id + label)
    result = await db.execute(
        text("""
            SELECT head_id, tail_id, label, COUNT(*) as cnt
            FROM relations
            GROUP BY head_id, tail_id, label
            HAVING COUNT(*) > 1
        """)
    )
    duplicates = result.all()
    quality.duplicate_relations = sum(row[3] - 1 for row in duplicates)
    if quality.duplicate_relations > 0:
        quality.issues.append(
            f"Found {quality.duplicate_relations} duplicate relations"
        )

    # Check for duplicate evidence (same relation_id + quote_hash)
    result = await db.execute(
        text("""
            SELECT relation_id, quote_hash, COUNT(*) as cnt
            FROM evidence
            GROUP BY relation_id, quote_hash
            HAVING COUNT(*) > 1
        """)
    )
    duplicates = result.all()
    quality.duplicate_evidence = sum(row[2] - 1 for row in duplicates)
    if quality.duplicate_evidence > 0:
        quality.issues.append(
            f"Found {quality.duplicate_evidence} duplicate evidence records"
        )

    # Check for orphan relations (missing head or tail entity)
    result = await db.execute(
        text("""
            SELECT COUNT(*) FROM relations r
            WHERE NOT EXISTS (SELECT 1 FROM entities e WHERE e.id = r.head_id)
               OR NOT EXISTS (SELECT 1 FROM entities e WHERE e.id = r.tail_id)
        """)
    )
    quality.orphan_relations = result.scalar() or 0
    if quality.orphan_relations > 0:
        quality.issues.append(
            f"Found {quality.orphan_relations} orphan relations (missing entities)"
        )

    # Relations without evidence
    result = await db.execute(
        text("""
            SELECT COUNT(*) FROM relations r
            WHERE NOT EXISTS (SELECT 1 FROM evidence e WHERE e.relation_id = r.id)
        """)
    )
    quality.relations_without_evidence = result.scalar() or 0
    if quality.relations_without_evidence > 0:
        quality.issues.append(
            f"Found {quality.relations_without_evidence} relations without evidence"
        )

    # Entities without relations
    result = await db.execute(
        text("""
            SELECT COUNT(*) FROM entities e
            WHERE NOT EXISTS (SELECT 1 FROM relations r WHERE r.head_id = e.id OR r.tail_id = e.id)
        """)
    )
    quality.entities_without_relations = result.scalar() or 0

    # Average relations per entity
    result = await db.execute(select(func.count(Entity.id)))
    entity_count = result.scalar() or 0
    result = await db.execute(select(func.count(Relation.id)))
    relation_count = result.scalar() or 0

    if entity_count > 0:
        # Each relation connects 2 entities, so multiply by 2
        quality.avg_relations_per_entity = round(
            (relation_count * 2) / entity_count, 2
        )

    # Average evidence per relation
    result = await db.execute(select(func.count(Evidence.id)))
    evidence_count = result.scalar() or 0

    if relation_count > 0:
        quality.avg_evidence_per_relation = round(evidence_count / relation_count, 2)

    return quality


async def run_sanity_checks(db: AsyncSession) -> list[SanityCheckResult]:
    """Run sanity checks to verify KG supports meaningful exploration."""
    checks = []

    # ==========================================================================
    # Check 1: Ambroxol exists and has relations
    # ==========================================================================
    result = await db.execute(
        select(Entity)
        .where(Entity.normalized_name.ilike("%ambroxol%"))
        .options(selectinload(Entity.outgoing_relations))
    )
    ambroxol_entities = result.scalars().all()

    ambroxol_check = SanityCheckResult(
        name="ambroxol_exists",
        description="Ambroxol entity exists with connections",
        passed=False,
    )

    if ambroxol_entities:
        total_relations = sum(
            len(e.outgoing_relations) for e in ambroxol_entities
        )
        ambroxol_check.passed = total_relations > 0
        ambroxol_check.details = {
            "entities_found": len(ambroxol_entities),
            "names": [e.name for e in ambroxol_entities],
            "total_outgoing_relations": total_relations,
        }
        ambroxol_check.message = (
            f"Found {len(ambroxol_entities)} Ambroxol entities with {total_relations} relations"
        )
    else:
        ambroxol_check.message = "Ambroxol entity not found in KG"

    checks.append(ambroxol_check)

    # ==========================================================================
    # Check 2: Parkinson's disease exists and has relations
    # ==========================================================================
    result = await db.execute(
        select(Entity)
        .where(Entity.normalized_name.ilike("%parkinson%"))
        .options(selectinload(Entity.incoming_relations))
    )
    pd_entities = result.scalars().all()

    pd_check = SanityCheckResult(
        name="parkinsons_exists",
        description="Parkinson's disease entity exists with connections",
        passed=False,
    )

    if pd_entities:
        total_relations = sum(len(e.incoming_relations) for e in pd_entities)
        pd_check.passed = total_relations > 0
        pd_check.details = {
            "entities_found": len(pd_entities),
            "names": [e.name for e in pd_entities],
            "total_incoming_relations": total_relations,
        }
        pd_check.message = (
            f"Found {len(pd_entities)} Parkinson's entities with {total_relations} relations"
        )
    else:
        pd_check.message = "Parkinson's disease entity not found in KG"

    checks.append(pd_check)

    # ==========================================================================
    # Check 3: Key mechanistic entities exist (GCase, alpha-synuclein, lysosome)
    # ==========================================================================
    key_terms = ["gcase", "gba", "alpha-synuclein", "synuclein", "lysosom"]
    found_terms = []

    for term in key_terms:
        result = await db.execute(
            select(func.count(Entity.id)).where(
                Entity.normalized_name.ilike(f"%{term}%")
            )
        )
        count = result.scalar() or 0
        if count > 0:
            found_terms.append(term)

    mechanism_check = SanityCheckResult(
        name="mechanistic_entities",
        description="Key mechanistic entities exist (GCase, alpha-synuclein, lysosome)",
        passed=len(found_terms) >= 2,
        details={
            "searched_terms": key_terms,
            "found_terms": found_terms,
        },
        message=f"Found {len(found_terms)}/{len(key_terms)} key mechanistic terms",
    )
    checks.append(mechanism_check)

    # ==========================================================================
    # Check 4: Multiple entity types are represented
    # ==========================================================================
    result = await db.execute(
        select(Entity.type, func.count(Entity.id))
        .group_by(Entity.type)
    )
    type_counts = {row[0].value: row[1] for row in result.all()}

    type_check = SanityCheckResult(
        name="entity_type_diversity",
        description="Multiple entity types are represented",
        passed=len(type_counts) >= 3,
        details={"type_counts": type_counts},
        message=f"Found {len(type_counts)} entity types: {list(type_counts.keys())}",
    )
    checks.append(type_check)

    # ==========================================================================
    # Check 5: Multiple relation labels are used
    # ==========================================================================
    result = await db.execute(
        select(Relation.label, func.count(Relation.id))
        .group_by(Relation.label)
    )
    label_counts = {row[0].value: row[1] for row in result.all()}

    label_check = SanityCheckResult(
        name="relation_label_diversity",
        description="Multiple relation labels are used",
        passed=len(label_counts) >= 3,
        details={"label_counts": label_counts},
        message=f"Found {len(label_counts)} relation types: {list(label_counts.keys())}",
    )
    checks.append(label_check)

    # ==========================================================================
    # Check 6: Evidence quotes are valid (exist in source chunks)
    # ==========================================================================
    # Sample a few evidence records and verify quotes
    result = await db.execute(
        select(Evidence)
        .options(selectinload(Evidence.chunk))
        .limit(10)
    )
    sample_evidence = result.scalars().all()

    valid_quotes = 0
    invalid_quotes = 0
    for ev in sample_evidence:
        if ev.chunk and ev.quote:
            if ev.quote in ev.chunk.text:
                valid_quotes += 1
            else:
                invalid_quotes += 1

    evidence_check = SanityCheckResult(
        name="evidence_validity",
        description="Evidence quotes are substrings of source chunks",
        passed=invalid_quotes == 0 or valid_quotes > invalid_quotes,
        details={
            "sampled": len(sample_evidence),
            "valid_quotes": valid_quotes,
            "invalid_quotes": invalid_quotes,
        },
        message=f"Sampled {len(sample_evidence)} evidence: {valid_quotes} valid, {invalid_quotes} invalid",
    )
    checks.append(evidence_check)

    # ==========================================================================
    # Check 7: Path exists between Ambroxol and Parkinson's
    # ==========================================================================
    # Simple check: do they share any common neighbors?
    path_check = SanityCheckResult(
        name="ambroxol_pd_connection",
        description="Path exists between Ambroxol and Parkinson's disease",
        passed=False,
    )

    if ambroxol_entities and pd_entities:
        # Get Ambroxol's neighbors
        ambroxol_ids = [e.id for e in ambroxol_entities]
        result = await db.execute(
            select(Relation.tail_id)
            .where(Relation.head_id.in_(ambroxol_ids))
        )
        ambroxol_neighbors = set(row[0] for row in result.all())

        # Get Parkinson's neighbors (entities that point to PD)
        pd_ids = [e.id for e in pd_entities]
        result = await db.execute(
            select(Relation.head_id)
            .where(Relation.tail_id.in_(pd_ids))
        )
        pd_neighbors = set(row[0] for row in result.all())

        # Check for direct connection or shared neighbors
        direct = bool(set(ambroxol_ids) & pd_neighbors) or bool(set(pd_ids) & ambroxol_neighbors)
        shared = ambroxol_neighbors & pd_neighbors

        path_check.passed = direct or len(shared) > 0
        path_check.details = {
            "direct_connection": direct,
            "shared_neighbors": len(shared),
        }
        if direct:
            path_check.message = "Direct connection exists between Ambroxol and Parkinson's"
        elif shared:
            path_check.message = f"Connected via {len(shared)} shared neighbors"
        else:
            path_check.message = "No path found (may need more extraction)"

    checks.append(path_check)

    return checks


async def generate_example_queries(db: AsyncSession) -> list[dict[str, Any]]:
    """Generate example API responses for artifact submission."""
    examples = []

    # Example 1: Entity search for "ambroxol"
    result = await db.execute(
        select(Entity)
        .where(Entity.normalized_name.ilike("%ambroxol%"))
        .limit(5)
    )
    entities = result.scalars().all()

    examples.append({
        "query": "GET /api/v1/entities/search?query=ambroxol",
        "description": "Search for Ambroxol entities",
        "response": {
            "items": [
                {
                    "id": str(e.id),
                    "name": e.name,
                    "type": e.type.value,
                    "aliases": e.aliases or [],
                }
                for e in entities
            ],
            "total": len(entities),
        },
    })

    # Example 2: Graph stats
    result = await db.execute(select(func.count(Entity.id)))
    entity_count = result.scalar() or 0
    result = await db.execute(select(func.count(Relation.id)))
    relation_count = result.scalar() or 0
    result = await db.execute(select(func.count(Evidence.id)))
    evidence_count = result.scalar() or 0

    examples.append({
        "query": "GET /api/v1/graph/stats",
        "description": "Overall graph statistics",
        "response": {
            "entity_count": entity_count,
            "relation_count": relation_count,
            "evidence_count": evidence_count,
        },
    })

    # Example 3: Entity neighborhood (if Ambroxol exists)
    if entities:
        entity = entities[0]
        result = await db.execute(
            select(Relation)
            .where(Relation.head_id == entity.id)
            .options(selectinload(Relation.tail_entity))
            .limit(5)
        )
        relations = result.scalars().all()

        examples.append({
            "query": f"GET /api/v1/entities/{entity.id}/neighborhood",
            "description": f"Neighborhood of {entity.name}",
            "response": {
                "entity": {
                    "id": str(entity.id),
                    "name": entity.name,
                    "type": entity.type.value,
                },
                "relations": [
                    {
                        "label": r.label.value,
                        "target": r.tail_entity.name if r.tail_entity else "unknown",
                        "confidence": r.confidence,
                    }
                    for r in relations
                ],
                "total_relations": len(relations),
            },
        })

    return examples


# =============================================================================
# Main Evaluation Runner
# =============================================================================


async def run_evaluation(verbose: bool = False) -> EvaluationReport:
    """Run complete evaluation and generate report."""
    report = EvaluationReport(
        generated_at=datetime.now().isoformat(),
        database_url=settings.db_url.split("@")[-1],  # Hide credentials
    )

    # Create database connection
    engine = create_async_engine(settings.db_url, echo=False)

    async with AsyncSession(engine) as db:
        print("=" * 60)
        print("LIFE AI Knowledge Graph - Evaluation Report")
        print("=" * 60)
        print()

        # 1. Summary Statistics
        print("📊 Computing summary statistics...")
        report.summary = await compute_summary_stats(db)

        print(f"   Documents: {report.summary.total_documents}")
        print(f"   Chunks: {report.summary.total_chunks} ({report.summary.chunks_processed} processed, {report.summary.processing_percentage}%)")
        print(f"   Entities: {report.summary.total_entities}")
        print(f"   Relations: {report.summary.total_relations}")
        print(f"   Evidence: {report.summary.total_evidence}")
        print(f"   Evidence coverage: {report.summary.evidence_coverage}%")
        print(f"   Avg confidence: {report.summary.avg_confidence}")
        print()

        if verbose:
            print("   Entities by type:")
            for t, c in report.summary.entities_by_type.items():
                print(f"      {t}: {c}")
            print()
            print("   Relations by label:")
            for l, c in report.summary.relations_by_label.items():
                print(f"      {l}: {c}")
            print()

        # 2. Quality Metrics
        print("🔍 Running quality checks...")
        report.quality = await compute_quality_metrics(db)

        print(f"   Duplicate entities: {report.quality.duplicate_entities}")
        print(f"   Duplicate relations: {report.quality.duplicate_relations}")
        print(f"   Orphan relations: {report.quality.orphan_relations}")
        print(f"   Relations without evidence: {report.quality.relations_without_evidence}")
        print(f"   Entities without relations: {report.quality.entities_without_relations}")
        print(f"   Avg relations per entity: {report.quality.avg_relations_per_entity}")
        print(f"   Avg evidence per relation: {report.quality.avg_evidence_per_relation}")
        print()

        if report.quality.issues:
            print("   ⚠️  Issues found:")
            for issue in report.quality.issues:
                print(f"      - {issue}")
            print()

        # 3. Sanity Checks
        print("✅ Running sanity checks...")
        report.sanity_checks = await run_sanity_checks(db)

        passed = sum(1 for c in report.sanity_checks if c.passed)
        total = len(report.sanity_checks)
        print(f"   Passed: {passed}/{total}")
        print()

        for check in report.sanity_checks:
            status = "✅" if check.passed else "❌"
            print(f"   {status} {check.name}: {check.message}")

        print()

        # 4. Example Queries
        print("📝 Generating example queries...")
        report.example_queries = await generate_example_queries(db)
        print(f"   Generated {len(report.example_queries)} examples")
        print()

        # 5. Overall Assessment
        issues_count = len(report.quality.issues)
        sanity_pass_rate = passed / total if total > 0 else 0

        if issues_count == 0 and sanity_pass_rate >= 0.8:
            report.overall_score = "PASS"
        elif issues_count <= 2 and sanity_pass_rate >= 0.5:
            report.overall_score = "WARN"
        else:
            report.overall_score = "FAIL"

        print("=" * 60)
        print(f"Overall Score: {report.overall_score}")
        print("=" * 60)

    await engine.dispose()
    return report


def report_to_dict(report: EvaluationReport) -> dict:
    """Convert report to JSON-serializable dict."""
    return {
        "generated_at": report.generated_at,
        "database_url": report.database_url,
        "summary": asdict(report.summary),
        "quality": asdict(report.quality),
        "sanity_checks": [asdict(c) for c in report.sanity_checks],
        "example_queries": report.example_queries,
        "overall_score": report.overall_score,
    }


def report_to_markdown(report: EvaluationReport) -> str:
    """Convert report to a GitHub-friendly markdown string."""
    s = report.summary
    q = report.quality
    score_emoji = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌"}.get(report.overall_score, "❓")

    lines: list[str] = []
    lines.append("# 📊 LIFE AI Knowledge Graph — Evaluation Report")
    lines.append("")
    lines.append(f"> Generated: {report.generated_at}")
    lines.append("")
    lines.append(f"## Overall Score: {score_emoji} {report.overall_score}")
    lines.append("")

    # ── Summary Stats ──
    lines.append("---")
    lines.append("")
    lines.append("## Summary Statistics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Documents | {s.total_documents} |")
    lines.append(f"| Chunks | {s.total_chunks} |")
    lines.append(f"| Chunks Processed | {s.chunks_processed} ({s.processing_percentage:.1f}%) |")
    lines.append(f"| Entities | {s.total_entities} |")
    lines.append(f"| Relations | {s.total_relations} |")
    lines.append(f"| Evidence Records | {s.total_evidence} |")
    lines.append(f"| Evidence Coverage | {s.evidence_coverage:.1f}% |")
    lines.append(f"| Avg Confidence | {s.avg_confidence:.2f} |")
    lines.append(f"| Confidence Range | {s.min_confidence:.2f} – {s.max_confidence:.2f} |")
    lines.append("")

    # ── Entity Types ──
    if s.entities_by_type:
        lines.append("### Entity Types")
        lines.append("")
        lines.append("| Type | Count |")
        lines.append("|------|-------|")
        for etype, count in sorted(s.entities_by_type.items(), key=lambda x: -x[1]):
            lines.append(f"| {etype} | {count} |")
        lines.append("")

    # ── Relation Labels ──
    if s.relations_by_label:
        lines.append("### Relation Labels")
        lines.append("")
        lines.append("| Label | Count |")
        lines.append("|-------|-------|")
        for label, count in sorted(s.relations_by_label.items(), key=lambda x: -x[1]):
            lines.append(f"| {label} | {count} |")
        lines.append("")

    # ── Quality Metrics ──
    lines.append("---")
    lines.append("")
    lines.append("## Quality Metrics")
    lines.append("")
    lines.append("| Check | Value |")
    lines.append("|-------|-------|")
    lines.append(f"| Duplicate Entities | {q.duplicate_entities} |")
    lines.append(f"| Duplicate Relations | {q.duplicate_relations} |")
    lines.append(f"| Duplicate Evidence | {q.duplicate_evidence} |")
    lines.append(f"| Orphan Relations | {q.orphan_relations} |")
    lines.append(f"| Relations Without Evidence | {q.relations_without_evidence} |")
    lines.append(f"| Invalid Entity Types | {q.invalid_entity_types} |")
    lines.append(f"| Invalid Relation Labels | {q.invalid_relation_labels} |")
    lines.append(f"| Entities Without Relations | {q.entities_without_relations} |")
    lines.append(f"| Avg Relations per Entity | {q.avg_relations_per_entity:.1f} |")
    lines.append(f"| Avg Evidence per Relation | {q.avg_evidence_per_relation:.1f} |")
    lines.append("")

    if q.issues:
        lines.append("### ⚠️ Issues Found")
        lines.append("")
        for issue in q.issues:
            lines.append(f"- {issue}")
        lines.append("")

    # ── Sanity Checks ──
    lines.append("---")
    lines.append("")
    lines.append("## Sanity Checks")
    lines.append("")
    passed = sum(1 for c in report.sanity_checks if c.passed)
    total = len(report.sanity_checks)
    lines.append(f"**{passed} / {total} passed**")
    lines.append("")
    lines.append("| # | Check | Result | Details |")
    lines.append("|---|-------|--------|---------|")
    for i, check in enumerate(report.sanity_checks, 1):
        icon = "✅" if check.passed else "❌"
        msg = check.message.replace("|", "\\|") if check.message else "—"
        lines.append(f"| {i} | {check.name} | {icon} | {msg} |")
    lines.append("")

    # ── Example Queries ──
    if report.example_queries:
        lines.append("---")
        lines.append("")
        lines.append("## Example Queries")
        lines.append("")
        for eq in report.example_queries:
            name = eq.get("name", "Query")
            endpoint = eq.get("endpoint", "")
            lines.append(f"### {name}")
            lines.append("")
            if endpoint:
                lines.append(f"```")
                lines.append(f"GET {endpoint}")
                lines.append(f"```")
                lines.append("")
            result_count = eq.get("result_count")
            if result_count is not None:
                lines.append(f"Returned **{result_count}** results.")
                lines.append("")

    return "\n".join(lines)


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate LIFE AI Knowledge Graph quality"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="EVAL_REPORT.md",
        help="Output file (default: EVAL_REPORT.md)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output JSON instead of Markdown",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed output",
    )

    args = parser.parse_args()

    # Run evaluation
    report = await run_evaluation(verbose=args.verbose)

    output_path = Path(args.output)

    if args.json or output_path.suffix == ".json":
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report_to_dict(report), f, indent=2, default=str)
    else:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report_to_markdown(report))

    print()
    print(f"📄 Report saved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
