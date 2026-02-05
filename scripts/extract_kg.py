#!/usr/bin/env python3
"""
Extract entities and relations from document chunks using LLM.

This script provides a command-line interface to:
1. Extract biomedical entities from text chunks
2. Extract relations between entities
3. Store results with evidence linking in the database

Usage:
    # Extract from all chunks (using mock LLM for testing)
    python scripts/extract_kg.py --provider mock

    # Extract using OpenAI
    python scripts/extract_kg.py --provider openai

    # Extract using Anthropic
    python scripts/extract_kg.py --provider anthropic

    # Limit number of chunks to process
    python scripts/extract_kg.py --limit 10 --provider mock

    # Dry run (extract but don't save)
    python scripts/extract_kg.py --dry-run --limit 5

Requirements:
    - Database must be running with documents and chunks
    - API keys configured in .env (for openai/anthropic)
"""

import argparse
import asyncio
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy import func, select  # noqa: E402

from src.core.config import settings  # noqa: E402
from src.core.logging import get_logger  # noqa: E402
from src.db import Chunk, Document, Entity, Evidence, Relation, get_db_context  # noqa: E402
from src.services.extraction import ExtractionService, extract_from_text  # noqa: E402
from src.services.llm_client import get_llm_client  # noqa: E402

logger = get_logger(__name__)


async def show_stats() -> dict:
    """Show current database statistics."""
    async with get_db_context() as db:
        # Count documents
        doc_count = await db.execute(select(func.count(Document.id)))
        total_docs = doc_count.scalar() or 0

        # Count chunks
        chunk_count = await db.execute(select(func.count(Chunk.id)))
        total_chunks = chunk_count.scalar() or 0

        # Count entities
        entity_count = await db.execute(select(func.count(Entity.id)))
        total_entities = entity_count.scalar() or 0

        # Count relations
        relation_count = await db.execute(select(func.count(Relation.id)))
        total_relations = relation_count.scalar() or 0

        # Count evidence records
        evidence_count = await db.execute(select(func.count(Evidence.id)))
        total_evidence = evidence_count.scalar() or 0

        # Count chunks with evidence (processed)
        processed_chunks = await db.execute(
            select(func.count(func.distinct(Evidence.chunk_id)))
        )
        with_evidence = processed_chunks.scalar() or 0

        stats = {
            "documents": total_docs,
            "chunks": total_chunks,
            "chunks_processed": with_evidence,
            "chunks_remaining": total_chunks - with_evidence,
            "entities": total_entities,
            "relations": total_relations,
            "evidence": total_evidence,
        }

        print("\nDatabase Statistics:")
        print(f"  Documents:          {stats['documents']}")
        print(f"  Chunks:             {stats['chunks']}")
        print(f"  Chunks processed:   {stats['chunks_processed']}")
        print(f"  Chunks remaining:   {stats['chunks_remaining']}")
        print(f"  Entities:           {stats['entities']}")
        print(f"  Relations:          {stats['relations']}")
        print(f"  Evidence records:   {stats['evidence']}")

        return stats


async def preview_extraction(provider: str, limit: int = 3) -> None:
    """Preview extraction on a few chunks without saving."""
    print(f"\nPreview extraction (provider={provider}):")
    print("=" * 60)

    async with get_db_context() as db:
        # Get a few chunks
        result = await db.execute(select(Chunk).limit(limit))
        chunks = result.scalars().all()

        if not chunks:
            print("No chunks found. Run chunking first.")
            return

        llm = get_llm_client(provider)

        async with llm:
            service = ExtractionService(db, llm)

            for chunk in chunks:
                print(f"\nChunk {chunk.id} (doc: {chunk.document_id}):")
                print(f"  Text: {chunk.text[:100]}...")
                print()

                try:
                    # Extract entities
                    entities = await service.extract_entities(chunk.text)
                    print(f"  Entities ({len(entities)}):")
                    for e in entities[:5]:
                        print(f"    - {e.name} ({e.type})")
                    if len(entities) > 5:
                        print(f"    ... and {len(entities) - 5} more")

                    # Extract relations
                    if entities:
                        relations = await service.extract_relations(chunk.text, entities)
                        print(f"  Relations ({len(relations)}):")
                        for r in relations[:3]:
                            print(f"    - {r.head} --[{r.label}]--> {r.tail}")
                            print(f"      Confidence: {r.confidence:.2f}")
                            print(f"      Evidence: \"{r.evidence[:60]}...\"")
                        if len(relations) > 3:
                            print(f"    ... and {len(relations) - 3} more")

                except Exception as e:
                    print(f"  Error: {e}")

                print("-" * 40)

    print("=" * 60)


async def run_extraction(
    provider: str,
    limit: int | None,
    skip_processed: bool,
    dry_run: bool,
    confidence_threshold: float,
    wait_seconds: int = 0,
    delay_seconds: int = 10,
) -> None:
    """Run the extraction pipeline."""
    start_time = datetime.now()

    print(f"\n{'='*60}")
    print("Knowledge Graph Extraction")
    print(f"{'='*60}")
    print(f"Provider:             {provider}")
    print(f"Limit:                {limit or 'all'}")
    print(f"Skip processed:       {skip_processed}")
    print(f"Confidence threshold: {confidence_threshold}")
    print(f"Dry run:              {dry_run}")
    print(f"Delay between chunks: {delay_seconds}s")
    print(f"{'='*60}\n")

    # Wait for rate limit cooldown if requested
    if wait_seconds > 0:
        print(f"Waiting {wait_seconds}s for rate limit cooldown...")
        import asyncio
        await asyncio.sleep(wait_seconds)
        print("Cooldown complete, starting extraction.\n")

    if dry_run:
        await preview_extraction(provider, limit=limit or 3)
        return

    # Progress callback
    def show_progress(processed: int, total: int) -> None:
        pct = (processed / total * 100) if total > 0 else 0
        print(f"\rProcessing chunks: {processed}/{total} ({pct:.0f}%)", end="", flush=True)

    llm = get_llm_client(provider)

    async with llm, get_db_context() as db:
        service = ExtractionService(
            db,
            llm,
            confidence_threshold=confidence_threshold,
            chunk_delay_seconds=delay_seconds,
        )

        result = await service.extract_all(
            limit=limit,
            skip_processed=skip_processed,
            progress_callback=show_progress,
        )

    print()  # New line after progress

    elapsed = (datetime.now() - start_time).total_seconds()

    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"Chunks processed:     {result.processed_chunks}")
    print(f"Entities extracted:   {result.total_entities}")
    print(f"Relations extracted:  {result.total_relations}")
    print(f"Errors:               {result.errors}")
    print(f"Time elapsed:         {elapsed:.1f}s")
    if result.processed_chunks > 0:
        print(f"Avg time per chunk:   {elapsed/result.processed_chunks:.2f}s")
    print(f"{'='*60}\n")

    # Show final stats
    await show_stats()


async def test_single_extraction(provider: str, text: str) -> None:
    """Test extraction on a single text input."""
    print(f"\nTest extraction (provider={provider}):")
    print(f"Text: {text[:100]}{'...' if len(text) > 100 else ''}")
    print("=" * 60)

    llm = get_llm_client(provider)

    async with llm:
        entities, relations = await extract_from_text(text, llm)

        print(f"\nEntities ({len(entities)}):")
        for e in entities:
            print(f"  - {e.name}")
            print(f"    Type: {e.type}")
            print(f"    Normalized: {e.normalized_name}")

        print(f"\nRelations ({len(relations)}):")
        for r in relations:
            print(f"  - {r.head} --[{r.label}]--> {r.tail}")
            print(f"    Confidence: {r.confidence:.2f}")
            print(f"    Evidence: \"{r.evidence}\"")

    print("=" * 60)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Extract entities and relations from document chunks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Show current statistics
    python scripts/extract_kg.py --stats

    # Test extraction on sample text
    python scripts/extract_kg.py --test "GBA1 mutations increase risk of Parkinson's disease"

    # Preview extraction (dry run)
    python scripts/extract_kg.py --dry-run --limit 3

    # Extract using mock LLM (for testing)
    python scripts/extract_kg.py --provider mock --limit 10

    # Extract using Google Gemini
    python scripts/extract_kg.py --provider gemini --limit 50

    # Extract all chunks
    python scripts/extract_kg.py --provider gemini

    # Re-process all chunks
    python scripts/extract_kg.py --provider gemini --no-skip
        """,
    )

    parser.add_argument(
        "--provider", "-p",
        type=str,
        default=settings.llm_extractor,
        choices=["gemini", "mock"],
        help=f"LLM provider to use (default: {settings.llm_extractor})",
    )

    parser.add_argument(
        "--limit", "-n",
        type=int,
        default=None,
        help="Maximum number of chunks to process",
    )

    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="Re-process chunks that already have extractions",
    )

    parser.add_argument(
        "--confidence", "-c",
        type=float,
        default=settings.extraction_confidence_threshold,
        help=f"Minimum confidence threshold (default: {settings.extraction_confidence_threshold})",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview extraction without saving to database",
    )

    parser.add_argument(
        "--wait", "-w",
        type=int,
        default=0,
        help="Wait N seconds before starting (for rate limit cooldown)",
    )

    parser.add_argument(
        "--delay", "-d",
        type=int,
        default=2,
        help="Delay in seconds between chunks (default: 2)",
    )

    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show database statistics only",
    )

    parser.add_argument(
        "--test",
        type=str,
        default=None,
        help="Test extraction on a single text input",
    )

    args = parser.parse_args()

    if args.stats:
        asyncio.run(show_stats())
        return

    if args.test:
        asyncio.run(test_single_extraction(args.provider, args.test))
        return

    asyncio.run(
        run_extraction(
            provider=args.provider,
            limit=args.limit,
            skip_processed=not args.no_skip,
            dry_run=args.dry_run,
            confidence_threshold=args.confidence,
            wait_seconds=args.wait,
            delay_seconds=args.delay,
        )
    )


if __name__ == "__main__":
    main()