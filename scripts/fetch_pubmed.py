#!/usr/bin/env python3
"""
Fetch PubMed articles and ingest them into the database.

This script provides a command-line interface to:
1. Search PubMed for Ambroxol-Parkinson's research articles
2. Fetch article metadata (title, abstract, authors, etc.)
3. Store documents in the database (idempotent)

Usage:
    # Fetch 50 articles with default query
    python scripts/fetch_pubmed.py --max-results 50

    # Custom query
    python scripts/fetch_pubmed.py --query "Ambroxol GCase" --max-results 100

    # With date filter
    python scripts/fetch_pubmed.py --min-date 2020 --max-results 100

    # Dry run (don't save to database)
    python scripts/fetch_pubmed.py --dry-run --max-results 10

Requirements:
    - Database must be running (docker-compose up -d postgres)
    - Migration must be applied (alembic upgrade head)
"""

import argparse
import asyncio
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy import select  # noqa: E402

from src.core.logging import get_logger  # noqa: E402
from src.db import Document, get_db_context  # noqa: E402
from src.services.pubmed import (  # noqa: E402
    DEFAULT_QUERY,
    PubMedArticle,
    PubMedClient,
    PubMedError,
)

logger = get_logger(__name__)


async def ingest_articles(
    articles: list[PubMedArticle],
    dry_run: bool = False,
) -> tuple[int, int, int]:
    """
    Ingest articles into the database.

    Uses upsert (INSERT ... ON CONFLICT) for idempotent ingestion.
    Articles with existing PMIDs are skipped.

    Args:
        articles: List of PubMedArticle objects to ingest
        dry_run: If True, don't actually save to database

    Returns:
        Tuple of (inserted_count, skipped_count, error_count)
    """
    if dry_run:
        logger.info("Dry run - not saving to database")
        return len(articles), 0, 0

    inserted = 0
    skipped = 0
    errors = 0

    async with get_db_context() as db:
        for article in articles:
            try:
                # Check if article already exists
                existing = await db.execute(
                    select(Document.id).where(Document.pmid == article.pmid)
                )
                if existing.scalar_one_or_none():
                    logger.debug("Article already exists", pmid=article.pmid)
                    skipped += 1
                    continue

                # Create document
                doc = Document(
                    pmid=article.pmid,
                    title=article.title,
                    abstract=article.abstract,
                    authors=[a.to_dict() for a in article.authors],
                    journal=article.journal,
                    publication_date=article.publication_date,
                    doi=article.doi,
                    mesh_terms=article.mesh_terms,
                    extra_data={
                        "keywords": article.keywords,
                        "publication_types": article.publication_types,
                    },
                )
                db.add(doc)
                await db.flush()
                inserted += 1

                logger.debug(
                    "Ingested article",
                    pmid=article.pmid,
                    title=article.title[:50],
                )

            except Exception as e:
                logger.error(
                    "Failed to ingest article",
                    pmid=article.pmid,
                    error=str(e),
                )
                errors += 1

        await db.commit()

    return inserted, skipped, errors


async def fetch_and_ingest(
    query: str,
    max_results: int,
    min_date: str | None,
    max_date: str | None,
    dry_run: bool,
    api_key: str | None,
) -> None:
    """
    Fetch articles from PubMed and ingest into database.

    Args:
        query: PubMed search query
        max_results: Maximum articles to fetch
        min_date: Minimum publication date (YYYY or YYYY/MM/DD)
        max_date: Maximum publication date (YYYY or YYYY/MM/DD)
        dry_run: If True, don't save to database
        api_key: Optional NCBI API key
    """
    start_time = datetime.now()

    print(f"\n{'='*60}")
    print("PubMed Article Fetcher")
    print(f"{'='*60}")
    print(f"Query: {query[:80]}{'...' if len(query) > 80 else ''}")
    print(f"Max results: {max_results}")
    if min_date:
        print(f"Min date: {min_date}")
    if max_date:
        print(f"Max date: {max_date}")
    print(f"Dry run: {dry_run}")
    print(f"{'='*60}\n")

    # Progress callback
    def show_progress(fetched: int, total: int) -> None:
        pct = (fetched / total * 100) if total > 0 else 0
        print(f"\rFetching articles: {fetched}/{total} ({pct:.0f}%)", end="", flush=True)

    try:
        async with PubMedClient(api_key=api_key) as client:
            # Search and fetch articles
            print("Searching PubMed...")
            articles = await client.search_and_fetch(
                query=query,
                max_results=max_results,
                min_date=min_date,
                max_date=max_date,
                progress_callback=show_progress,
            )
            print()  # New line after progress

            if not articles:
                print("\nNo articles found matching the query.")
                return

            # Filter articles with abstracts
            articles_with_abstract = [a for a in articles if a.has_abstract]
            print(f"\nFound {len(articles)} articles ({len(articles_with_abstract)} with abstracts)")

            # Show sample articles
            print("\nSample articles:")
            for article in articles[:5]:
                abstract_status = "✓" if article.has_abstract else "✗"
                print(f"  [{abstract_status}] {article.pmid}: {article.title[:60]}...")

            if len(articles) > 5:
                print(f"  ... and {len(articles) - 5} more")

            # Ingest into database
            print("\nIngesting articles...")
            inserted, skipped, errors = await ingest_articles(articles, dry_run=dry_run)

            # Summary
            elapsed = (datetime.now() - start_time).total_seconds()
            print(f"\n{'='*60}")
            print("Summary")
            print(f"{'='*60}")
            print(f"Total fetched:    {len(articles)}")
            print(f"With abstracts:   {len(articles_with_abstract)}")
            print(f"Inserted:         {inserted}")
            print(f"Skipped (exists): {skipped}")
            print(f"Errors:           {errors}")
            print(f"Time elapsed:     {elapsed:.1f}s")
            print(f"{'='*60}\n")

    except PubMedError as e:
        logger.error("PubMed error", error=str(e))
        print(f"\nError: {e}")
        sys.exit(1)
    except Exception as e:
        logger.exception("Unexpected error")
        print(f"\nUnexpected error: {e}")
        sys.exit(1)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Fetch PubMed articles and ingest into the database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Fetch 50 articles with default Ambroxol-Parkinson query
    python scripts/fetch_pubmed.py --max-results 50

    # Custom query
    python scripts/fetch_pubmed.py --query "GBA1 mutation Parkinson" --max-results 100

    # Filter by date
    python scripts/fetch_pubmed.py --min-date 2020 --max-results 100

    # Dry run (preview without saving)
    python scripts/fetch_pubmed.py --dry-run --max-results 10
        """,
    )

    parser.add_argument(
        "--query", "-q",
        type=str,
        default=DEFAULT_QUERY,
        help="PubMed search query (default: Ambroxol-Parkinson query)",
    )

    parser.add_argument(
        "--max-results", "-n",
        type=int,
        default=100,
        help="Maximum number of articles to fetch (default: 100)",
    )

    parser.add_argument(
        "--min-date",
        type=str,
        default=None,
        help="Minimum publication date (YYYY or YYYY/MM/DD)",
    )

    parser.add_argument(
        "--max-date",
        type=str,
        default=None,
        help="Maximum publication date (YYYY or YYYY/MM/DD)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch articles but don't save to database",
    )

    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="NCBI API key for higher rate limits",
    )

    args = parser.parse_args()

    # Run async main
    asyncio.run(
        fetch_and_ingest(
            query=args.query,
            max_results=args.max_results,
            min_date=args.min_date,
            max_date=args.max_date,
            dry_run=args.dry_run,
            api_key=args.api_key,
        )
    )


if __name__ == "__main__":
    main()
