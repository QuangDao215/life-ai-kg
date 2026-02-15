"""
Audit evidence records for quote validity.

Checks every evidence record's quote against its source chunk text and
classifies each into: exact match, case-insensitive match, fuzzy match,
or no match (hallucinated).

Usage:
    python scripts/audit_evidence.py                  # Full report
    python scripts/audit_evidence.py --limit 100      # Audit first 100 records
    python scripts/audit_evidence.py --export          # Save detailed CSV report
    python scripts/audit_evidence.py --show-invalid    # Print invalid quotes
"""

import argparse
import asyncio
import csv
import os
from dataclasses import dataclass
from difflib import SequenceMatcher

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from src.core.config import settings
from src.core.logging import get_logger
from src.db.models import Evidence, Chunk

logger = get_logger(__name__)


# ============================================================================
# Classification
# ============================================================================

# Fuzzy match threshold: ratio above this is considered a fuzzy match
FUZZY_THRESHOLD = 0.80


@dataclass
class AuditResult:
    """Result of auditing a single evidence record."""

    evidence_id: str
    relation_id: str
    chunk_id: str
    category: str           # exact | case_insensitive | fuzzy | no_match
    similarity: float       # 0.0 to 1.0
    quote_length: int
    chunk_length: int
    quote_preview: str      # first 80 chars
    best_substring: str     # best matching substring from chunk (for fuzzy)


def classify_quote(quote: str, chunk_text: str) -> AuditResult:
    """
    Classify a quote against its source chunk text.

    Returns an AuditResult with the classification category and similarity.
    """
    base = {
        "quote_length": len(quote),
        "chunk_length": len(chunk_text),
        "quote_preview": quote[:80].replace("\n", " "),
    }

    # 1. Exact match
    if quote in chunk_text:
        return AuditResult(
            evidence_id="", relation_id="", chunk_id="",
            category="exact",
            similarity=1.0,
            best_substring=quote,
            **base,
        )

    # 2. Case-insensitive match
    if quote.lower() in chunk_text.lower():
        return AuditResult(
            evidence_id="", relation_id="", chunk_id="",
            category="case_insensitive",
            similarity=1.0,
            best_substring=quote,
            **base,
        )

    # 3. Fuzzy match — find best matching substring in chunk
    best_ratio, best_sub = find_best_substring(quote, chunk_text)

    if best_ratio >= FUZZY_THRESHOLD:
        return AuditResult(
            evidence_id="", relation_id="", chunk_id="",
            category="fuzzy",
            similarity=best_ratio,
            best_substring=best_sub,
            **base,
        )

    # 4. No match
    return AuditResult(
        evidence_id="", relation_id="", chunk_id="",
        category="no_match",
        similarity=best_ratio,
        best_substring=best_sub,
        **base,
    )


def find_best_substring(quote: str, text: str) -> tuple[float, str]:
    """
    Find the substring of `text` that best matches `quote`.

    Uses a sliding window approach with SequenceMatcher for efficiency.
    Returns (similarity_ratio, best_matching_substring).
    """
    quote_lower = quote.lower()
    text_lower = text.lower()
    quote_len = len(quote_lower)

    if not quote_lower or not text_lower:
        return 0.0, ""

    # For very short quotes, just compare against full text
    if quote_len < 20:
        ratio = SequenceMatcher(None, quote_lower, text_lower).ratio()
        return ratio, text[:len(quote)]

    best_ratio = 0.0
    best_start = 0

    # Sliding window: try windows of similar size to the quote
    # Check windows from 0.8x to 1.2x the quote length
    min_window = max(10, int(quote_len * 0.8))
    max_window = min(len(text_lower), int(quote_len * 1.2))

    # Step size — skip some positions for performance on long texts
    step = max(1, quote_len // 10)

    for window_size in range(min_window, max_window + 1, max(1, (max_window - min_window) // 3)):
        for start in range(0, len(text_lower) - window_size + 1, step):
            candidate = text_lower[start:start + window_size]
            ratio = SequenceMatcher(None, quote_lower, candidate).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_start = start

    # Refine: search around the best position with step=1
    if step > 1 and best_ratio > 0.5:
        refine_start = max(0, best_start - step)
        refine_end = min(len(text_lower), best_start + step + quote_len)
        for window_size in range(min_window, max_window + 1, max(1, (max_window - min_window) // 3)):
            for start in range(refine_start, min(refine_end, len(text_lower) - window_size + 1)):
                candidate = text_lower[start:start + window_size]
                ratio = SequenceMatcher(None, quote_lower, candidate).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_start = start

    best_sub = text[best_start:best_start + quote_len]
    return best_ratio, best_sub


# ============================================================================
# Database Audit
# ============================================================================

async def run_audit(
    limit: int | None = None,
    export_path: str | None = None,
    show_invalid: bool = False,
) -> dict:
    """
    Audit all evidence records against their source chunks.

    Returns summary statistics dict.
    """
    engine = create_async_engine(settings.db_url, echo=False)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    try:
        async with session_factory() as db:
            # Count total evidence
            total_result = await db.execute(select(func.count(Evidence.id)))
            total_evidence = total_result.scalar()

            # Fetch evidence with chunk text via join
            query = (
                select(Evidence, Chunk.text)
                .join(Chunk, Evidence.chunk_id == Chunk.id)
                .order_by(Evidence.created_at)
            )
            if limit:
                query = query.limit(limit)

            result = await db.execute(query)
            rows = result.all()

        # Classify each evidence record
        results: list[AuditResult] = []
        counts = {"exact": 0, "case_insensitive": 0, "fuzzy": 0, "no_match": 0}

        print(f"\nAuditing {len(rows)} evidence records...\n")

        for i, (evidence, chunk_text) in enumerate(rows):
            audit = classify_quote(evidence.quote, chunk_text)
            audit.evidence_id = str(evidence.id)
            audit.relation_id = str(evidence.relation_id)
            audit.chunk_id = str(evidence.chunk_id)
            results.append(audit)
            counts[audit.category] += 1

            # Progress
            if (i + 1) % 500 == 0 or (i + 1) == len(rows):
                print(f"  Processed {i + 1}/{len(rows)} records...", flush=True)

        audited = len(results)
        valid = counts["exact"] + counts["case_insensitive"] + counts["fuzzy"]
        invalid = counts["no_match"]

        # Print summary
        print("\n" + "=" * 60)
        print("Evidence Audit Report")
        print("=" * 60)
        print(f"\n  Total evidence in database:  {total_evidence}")
        print(f"  Audited:                     {audited}")
        print(f"\n  Results:")
        print(f"    Exact match:               {counts['exact']:>5}  ({_pct(counts['exact'], audited)})")
        print(f"    Case-insensitive match:    {counts['case_insensitive']:>5}  ({_pct(counts['case_insensitive'], audited)})")
        print(f"    Fuzzy match (>{FUZZY_THRESHOLD:.0%}):      {counts['fuzzy']:>5}  ({_pct(counts['fuzzy'], audited)})")
        print(f"    No match (hallucinated):   {counts['no_match']:>5}  ({_pct(counts['no_match'], audited)})")
        print(f"\n  Valid (exact + case + fuzzy): {valid:>5}  ({_pct(valid, audited)})")
        print(f"  Invalid (no match):          {invalid:>5}  ({_pct(invalid, audited)})")

        # Show invalid records if requested
        if show_invalid and invalid > 0:
            print(f"\n{'=' * 60}")
            print("Invalid Evidence Records (No Match)")
            print("=" * 60)
            for r in results:
                if r.category == "no_match":
                    print(f"\n  Evidence ID: {r.evidence_id}")
                    print(f"  Chunk ID:    {r.chunk_id}")
                    print(f"  Similarity:  {r.similarity:.2%}")
                    print(f"  Quote:       \"{r.quote_preview}...\"")
                    print(f"  Best match:  \"{r.best_substring[:80]}...\"")

        # Show fuzzy matches summary
        fuzzy_results = [r for r in results if r.category == "fuzzy"]
        if fuzzy_results:
            avg_sim = sum(r.similarity for r in fuzzy_results) / len(fuzzy_results)
            min_sim = min(r.similarity for r in fuzzy_results)
            print(f"\n  Fuzzy match details:")
            print(f"    Average similarity: {avg_sim:.2%}")
            print(f"    Minimum similarity: {min_sim:.2%}")

        # Export CSV if requested
        if export_path:
            _export_csv(results, export_path)
            print(f"\n  Detailed report exported to: {export_path}")

        print()

        return {
            "total_evidence": total_evidence,
            "audited": audited,
            "exact": counts["exact"],
            "case_insensitive": counts["case_insensitive"],
            "fuzzy": counts["fuzzy"],
            "no_match": counts["no_match"],
            "valid": valid,
            "invalid": invalid,
            "validity_rate": valid / audited if audited > 0 else 0,
        }

    finally:
        await engine.dispose()


def _pct(count: int, total: int) -> str:
    """Format as percentage string."""
    if total == 0:
        return "0.0%"
    return f"{count / total * 100:.1f}%"


def _export_csv(results: list[AuditResult], path: str) -> None:
    """Export audit results to CSV."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "evidence_id", "relation_id", "chunk_id", "category",
            "similarity", "quote_length", "chunk_length", "quote_preview",
        ])
        for r in results:
            writer.writerow([
                r.evidence_id, r.relation_id, r.chunk_id, r.category,
                f"{r.similarity:.4f}", r.quote_length, r.chunk_length,
                r.quote_preview,
            ])


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Audit evidence quote validity")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Max evidence records to audit (default: all)",
    )
    parser.add_argument(
        "--export", action="store_true",
        help="Export detailed CSV report to demo_output/evidence_audit.csv",
    )
    parser.add_argument(
        "--export-path", type=str, default="demo_output/evidence_audit.csv",
        help="Path for CSV export (default: demo_output/evidence_audit.csv)",
    )
    parser.add_argument(
        "--show-invalid", action="store_true",
        help="Print details of invalid (hallucinated) evidence records",
    )
    args = parser.parse_args()

    export_path = args.export_path if args.export else None

    asyncio.run(run_audit(
        limit=args.limit,
        export_path=export_path,
        show_invalid=args.show_invalid,
    ))


if __name__ == "__main__":
    main()