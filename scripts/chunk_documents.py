#!/usr/bin/env python3
"""
Chunk documents in the database.

This script provides a command-line interface to:
1. Split document abstracts into chunks for LLM processing
2. Track character offsets for evidence provenance
3. Store chunks in the database

Usage:
    # Chunk all documents without existing chunks
    python scripts/chunk_documents.py

    # Re-chunk all documents (replace existing)
    python scripts/chunk_documents.py --replace

    # Custom chunk size
    python scripts/chunk_documents.py --chunk-size 500 --overlap 100

    # Dry run (preview without saving)
    python scripts/chunk_documents.py --dry-run

Requirements:
    - Database must be running (docker-compose up -d postgres)
    - Documents must be ingested (python scripts/fetch_pubmed.py)
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
from src.db import Chunk, Document, get_db_context  # noqa: E402
from src.services.chunking import ChunkingService, chunk_text  # noqa: E402

logger = get_logger(__name__)


async def show_stats() -> None:
    """Show current database statistics."""
    async with get_db_context() as db:
        # Count documents
        doc_count = await db.execute(select(func.count(Document.id)))
        total_docs = doc_count.scalar() or 0

        # Count documents with abstracts
        docs_with_abstract = await db.execute(
            select(func.count(Document.id)).where(Document.abstract.isnot(None))
        )
        with_abstract = docs_with_abstract.scalar() or 0

        # Count chunks
        chunk_count = await db.execute(select(func.count(Chunk.id)))
        total_chunks = chunk_count.scalar() or 0

        # Count documents with chunks
        docs_with_chunks = await db.execute(
            select(func.count(func.distinct(Chunk.document_id)))
        )
        with_chunks = docs_with_chunks.scalar() or 0

        # Documents needing chunking
        needs_chunking = with_abstract - with_chunks

        print("\nDatabase Statistics:")
        print(f"  Total documents:        {total_docs}")
        print(f"  With abstracts:         {with_abstract}")
        print(f"  Already chunked:        {with_chunks}")
        print(f"  Needing chunking:       {needs_chunking}")
        print(f"  Total chunks:           {total_chunks}")

        return needs_chunking


async def preview_chunking(
    chunk_size: int,
    chunk_overlap: int,
    limit: int = 3,
) -> None:
    """Preview chunking on a few documents."""
    async with get_db_context() as db:
        # Get a few documents with abstracts
        result = await db.execute(
            select(Document)
            .where(Document.abstract.isnot(None))
            .limit(limit)
        )
        documents = result.scalars().all()

        print(f"\nPreview chunking (chunk_size={chunk_size}, overlap={chunk_overlap}):")
        print("=" * 60)

        for doc in documents:
            print(f"\nPMID: {doc.pmid}")
            print(f"Title: {doc.title[:60]}...")
            print(f"Abstract length: {len(doc.abstract)} chars")

            chunks = list(chunk_text(
                doc.abstract,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            ))

            print(f"Chunks: {len(chunks)}")
            for chunk in chunks[:3]:  # Show first 3 chunks
                print(f"  [{chunk.chunk_index}] {chunk.start_offset}-{chunk.end_offset} "
                      f"({chunk.token_count} tokens): {chunk.text[:50]}...")

            if len(chunks) > 3:
                print(f"  ... and {len(chunks) - 3} more chunks")

        print("=" * 60)


async def run_chunking(
    chunk_size: int,
    chunk_overlap: int,
    replace_existing: bool,
    dry_run: bool,
) -> None:
    """
    Run the chunking process.

    Args:
        chunk_size: Target chunk size in characters
        chunk_overlap: Overlap between chunks
        replace_existing: Whether to re-chunk existing documents
        dry_run: If True, don't save to database
    """
    start_time = datetime.now()

    print(f"\n{'='*60}")
    print("Document Chunking")
    print(f"{'='*60}")
    print(f"Chunk size:       {chunk_size} chars")
    print(f"Chunk overlap:    {chunk_overlap} chars")
    print(f"Replace existing: {replace_existing}")
    print(f"Dry run:          {dry_run}")
    print(f"{'='*60}\n")

    if dry_run:
        await preview_chunking(chunk_size, chunk_overlap)
        needs_chunking = await show_stats()
        print(f"\nDry run complete. {needs_chunking} documents would be chunked.")
        return

    # Progress callback
    def show_progress(processed: int, total: int) -> None:
        pct = (processed / total * 100) if total > 0 else 0
        print(f"\rChunking documents: {processed}/{total} ({pct:.0f}%)", end="", flush=True)

    async with get_db_context() as db:
        service = ChunkingService(
            db,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        results = await service.chunk_all_documents(
            replace_existing=replace_existing,
            progress_callback=show_progress,
        )

    print()  # New line after progress

    # Calculate statistics
    total_docs = len(results)
    total_chunks = sum(r.chunk_count for r in results)
    total_tokens = sum(r.total_tokens for r in results)
    docs_with_chunks = sum(1 for r in results if r.chunk_count > 0)
    avg_chunks = total_chunks / docs_with_chunks if docs_with_chunks > 0 else 0

    elapsed = (datetime.now() - start_time).total_seconds()

    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"Documents processed:  {total_docs}")
    print(f"Documents chunked:    {docs_with_chunks}")
    print(f"Total chunks created: {total_chunks}")
    print(f"Total tokens:         {total_tokens:,}")
    print(f"Avg chunks/document:  {avg_chunks:.1f}")
    print(f"Time elapsed:         {elapsed:.1f}s")
    print(f"{'='*60}\n")

    # Show final stats
    await show_stats()


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Chunk documents for LLM processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Chunk all documents without existing chunks
    python scripts/chunk_documents.py

    # Re-chunk all documents
    python scripts/chunk_documents.py --replace

    # Custom chunk size (smaller for faster LLM processing)
    python scripts/chunk_documents.py --chunk-size 500 --overlap 100

    # Preview without saving
    python scripts/chunk_documents.py --dry-run

    # Show statistics only
    python scripts/chunk_documents.py --stats
        """,
    )

    parser.add_argument(
        "--chunk-size", "-s",
        type=int,
        default=settings.chunk_size,
        help=f"Target chunk size in characters (default: {settings.chunk_size})",
    )

    parser.add_argument(
        "--overlap", "-o",
        type=int,
        default=settings.chunk_overlap,
        help=f"Overlap between chunks in characters (default: {settings.chunk_overlap})",
    )

    parser.add_argument(
        "--replace", "-r",
        action="store_true",
        help="Replace existing chunks (re-chunk all documents)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview chunking without saving to database",
    )

    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show database statistics only",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.overlap >= args.chunk_size:
        print(f"Error: Overlap ({args.overlap}) must be less than chunk size ({args.chunk_size})")
        sys.exit(1)

    if args.stats:
        asyncio.run(show_stats())
        return

    asyncio.run(
        run_chunking(
            chunk_size=args.chunk_size,
            chunk_overlap=args.overlap,
            replace_existing=args.replace,
            dry_run=args.dry_run,
        )
    )


if __name__ == "__main__":
    main()
