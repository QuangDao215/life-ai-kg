"""
Repair fuzzy-match evidence quotes using sentence-aligned matching.

Instead of replacing quotes with raw sliding-window substrings (which can
cut mid-word), this script:

1. Splits each source chunk into sentences
2. Finds the approximate match region using fuzzy matching
3. Expands to cover complete sentence(s) that overlap the match region
4. Replaces the quote with those full sentence(s)

This ensures every repaired quote is a clean, complete sentence.

Usage:
    python scripts/repair_evidence.py                  # Dry run (show what would change)
    python scripts/repair_evidence.py --execute        # Apply repairs
    python scripts/repair_evidence.py --execute --export  # Apply + save CSV report
    python scripts/repair_evidence.py --threshold 0.85 # Custom similarity threshold
    python scripts/repair_evidence.py -v               # Verbose: show old vs new quotes
"""

import argparse
import asyncio
import csv
import hashlib
import os
import re
from dataclasses import dataclass
from difflib import SequenceMatcher

from sqlalchemy import select, func, update
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from src.core.config import settings
from src.core.logging import get_logger
from src.db.models import Evidence, Chunk

logger = get_logger(__name__)


# ============================================================================
# Configuration
# ============================================================================

DEFAULT_FUZZY_THRESHOLD = 0.80


# ============================================================================
# Data structures
# ============================================================================

@dataclass
class SentenceSpan:
    """A sentence with its character offsets in the parent text."""
    text: str
    start: int
    end: int


@dataclass
class RepairCandidate:
    """A fuzzy-match evidence record that can be repaired."""
    evidence_id: str
    relation_id: str
    chunk_id: str

    old_quote: str
    new_quote: str
    similarity: float           # fuzzy similarity of old quote vs chunk region
    sentence_count: int         # how many sentences the new quote spans

    old_hash: str
    new_hash: str
    new_quote_start: int
    new_quote_end: int

    @property
    def quote_changed(self) -> bool:
        return self.old_quote != self.new_quote


# ============================================================================
# Sentence splitting
# ============================================================================

# Abbreviations common in biomedical text that should NOT trigger a split
ABBREVIATIONS = {
    "e.g", "i.e", "et al", "vs", "fig", "figs", "ref", "refs",
    "vol", "no", "approx", "ca", "dept", "dr", "mr", "mrs", "ms",
    "prof", "inc", "ltd", "co", "corp", "jr", "sr", "st",
}

# Regex: split on period/question/exclamation followed by whitespace + uppercase
# But not after known abbreviations or decimal numbers
SENTENCE_SPLIT_RE = re.compile(
    r'(?<=[.!?])'       # lookbehind: sentence-ending punctuation
    r'(?:\s+)'          # whitespace between sentences
    r'(?=[A-Z([])'      # lookahead: next sentence starts with uppercase or bracket
)


def split_sentences(text: str) -> list[SentenceSpan]:
    """
    Split text into sentences, preserving character offsets.

    Handles biomedical abbreviations (e.g., et al., vs., fig.)
    to avoid false splits.
    """
    if not text.strip():
        return []

    # Find all potential split points
    splits = list(SENTENCE_SPLIT_RE.finditer(text))

    # Filter out splits that occur right after abbreviations
    valid_splits = []
    for match in splits:
        pos = match.start()
        before = text[:pos].rstrip()
        words = before.split()
        if words:
            last_word = words[-1].rstrip(".!?").lower()
            if last_word in ABBREVIATIONS:
                continue
            if re.match(r'\d+\.\d', before[-10:]):
                continue
        valid_splits.append(match)

    # Build sentence spans from split points
    sentences: list[SentenceSpan] = []
    prev_end = 0

    for match in valid_splits:
        sent_end = match.start()
        next_start = match.end()

        sent_text = text[prev_end:sent_end].strip()
        if sent_text:
            sentences.append(SentenceSpan(
                text=sent_text,
                start=prev_end,
                end=sent_end,
            ))
        prev_end = next_start

    # Last sentence
    remaining = text[prev_end:].strip()
    if remaining:
        sentences.append(SentenceSpan(
            text=remaining,
            start=prev_end,
            end=len(text),
        ))

    # Fallback: entire text as one sentence
    if not sentences:
        sentences.append(SentenceSpan(text=text.strip(), start=0, end=len(text)))

    return sentences


# ============================================================================
# Matching logic
# ============================================================================

def find_approximate_region(quote: str, text: str) -> tuple[float, int, int]:
    """
    Find the approximate character region in `text` that matches `quote`.

    Returns (similarity, region_start, region_end).
    Used only to locate the region — NOT for the final replacement text.
    """
    quote_lower = quote.lower()
    text_lower = text.lower()
    quote_len = len(quote_lower)

    if not quote_lower or not text_lower:
        return 0.0, 0, 0

    if quote_len < 20:
        ratio = SequenceMatcher(None, quote_lower, text_lower).ratio()
        return ratio, 0, min(len(text), quote_len)

    best_ratio = 0.0
    best_start = 0
    best_window = quote_len

    min_window = max(10, int(quote_len * 0.8))
    max_window = min(len(text_lower), int(quote_len * 1.2))
    step = max(1, quote_len // 10)

    for window_size in range(min_window, max_window + 1, max(1, (max_window - min_window) // 3)):
        for start in range(0, len(text_lower) - window_size + 1, step):
            candidate = text_lower[start:start + window_size]
            ratio = SequenceMatcher(None, quote_lower, candidate).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_start = start
                best_window = window_size

    # Refine around best position
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
                    best_window = window_size

    return best_ratio, best_start, best_start + best_window


def find_best_sentences(
    quote: str,
    chunk_text: str,
    sentences: list[SentenceSpan],
) -> tuple[str, float, int, int, int]:
    """
    Find the sentence(s) in the chunk that best match the quote.

    Strategy:
    1. Use fuzzy matching to locate the approximate region
    2. Find all sentences that overlap with that region
    3. Among overlapping sentence combinations, pick the one with highest
       similarity to the original quote

    Returns (best_quote, similarity, start_offset, end_offset, num_sentences).
    """
    if not sentences:
        return "", 0.0, 0, 0, 0

    # Step 1: Find approximate match region
    region_sim, region_start, region_end = find_approximate_region(quote, chunk_text)

    if region_sim < 0.5:
        return "", region_sim, 0, 0, 0

    # Step 2: Find sentences that overlap with the match region
    overlapping = []
    margin = 20  # chars of slack
    for i, sent in enumerate(sentences):
        if sent.end > (region_start - margin) and sent.start < (region_end + margin):
            overlapping.append(i)

    if not overlapping:
        # Fallback: find the single closest sentence
        min_dist = float("inf")
        closest_idx = 0
        region_mid = (region_start + region_end) / 2
        for i, sent in enumerate(sentences):
            sent_mid = (sent.start + sent.end) / 2
            dist = abs(sent_mid - region_mid)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
        overlapping = [closest_idx]

    # Step 3: Try contiguous subsets of overlapping sentences
    best_quote = ""
    best_sim = 0.0
    best_start = 0
    best_end = 0
    best_count = 0

    min_idx = min(overlapping)
    max_idx = max(overlapping)

    for window_start in range(min_idx, max_idx + 1):
        for window_end in range(window_start, min(max_idx + 1, window_start + 3)):
            span_start = sentences[window_start].start
            span_end = sentences[window_end].end
            candidate = chunk_text[span_start:span_end].strip()

            if not candidate:
                continue

            sim = SequenceMatcher(None, quote.lower(), candidate.lower()).ratio()

            # Penalize candidates much longer than original quote
            length_ratio = len(candidate) / max(len(quote), 1)
            if length_ratio > 2.0:
                sim *= 0.8
            elif length_ratio > 3.0:
                sim *= 0.6

            if sim > best_sim:
                best_sim = sim
                best_quote = candidate
                best_start = span_start
                best_end = span_end
                best_count = window_end - window_start + 1

    return best_quote, best_sim, best_start, best_end, best_count


def is_exact_match(quote: str, chunk_text: str) -> bool:
    """Check if quote exists verbatim in chunk (exact or case-insensitive)."""
    return quote in chunk_text or quote.lower() in chunk_text.lower()


def compute_quote_hash(quote: str) -> str:
    """Compute SHA-256 hash of a quote."""
    return hashlib.sha256(quote.encode("utf-8")).hexdigest()


# ============================================================================
# Repair pipeline
# ============================================================================

async def find_repair_candidates(
    session_factory,
    threshold: float = DEFAULT_FUZZY_THRESHOLD,
) -> list[RepairCandidate]:
    """
    Scan all evidence records and identify fuzzy matches repairable
    via sentence-aligned replacement.
    """
    async with session_factory() as db:
        query = (
            select(Evidence, Chunk.text)
            .join(Chunk, Evidence.chunk_id == Chunk.id)
            .order_by(Evidence.created_at)
        )
        result = await db.execute(query)
        rows = result.all()

    candidates: list[RepairCandidate] = []
    skipped_exact = 0
    skipped_low = 0

    print(f"\nScanning {len(rows)} evidence records for repairable fuzzy matches...\n")

    for i, (evidence, chunk_text) in enumerate(rows):
        # Skip exact matches
        if is_exact_match(evidence.quote, chunk_text):
            skipped_exact += 1
            if (i + 1) % 500 == 0:
                print(f"  Scanned {i + 1}/{len(rows)}... "
                      f"({len(candidates)} candidates)", flush=True)
            continue

        # Split chunk into sentences
        sentences = split_sentences(chunk_text)

        # Find best sentence-aligned match
        new_quote, sim, start, end, sent_count = find_best_sentences(
            evidence.quote, chunk_text, sentences,
        )

        if sim >= threshold and new_quote:
            new_hash = compute_quote_hash(new_quote)
            candidates.append(RepairCandidate(
                evidence_id=str(evidence.id),
                relation_id=str(evidence.relation_id),
                chunk_id=str(evidence.chunk_id),
                old_quote=evidence.quote,
                new_quote=new_quote,
                similarity=sim,
                sentence_count=sent_count,
                old_hash=evidence.quote_hash if hasattr(evidence, "quote_hash") else "",
                new_hash=new_hash,
                new_quote_start=start,
                new_quote_end=end,
            ))
        else:
            skipped_low += 1

        if (i + 1) % 500 == 0 or (i + 1) == len(rows):
            print(f"  Scanned {i + 1}/{len(rows)}... "
                  f"({len(candidates)} candidates)", flush=True)

    print(f"\n  Exact matches (no repair needed): {skipped_exact}")
    print(f"  Below threshold (not repairable): {skipped_low}")
    print(f"  Repair candidates found:          {len(candidates)}")

    return candidates


async def apply_repairs(
    session_factory,
    candidates: list[RepairCandidate],
) -> int:
    """Apply sentence-aligned repairs to the database."""
    repaired = 0

    async with session_factory() as db:
        for candidate in candidates:
            if not candidate.quote_changed:
                continue

            try:
                stmt = (
                    update(Evidence)
                    .where(Evidence.id == candidate.evidence_id)
                    .values(
                        quote=candidate.new_quote,
                        quote_hash=candidate.new_hash,
                        quote_start=candidate.new_quote_start,
                        quote_end=candidate.new_quote_end,
                    )
                )
                await db.execute(stmt)
            except Exception:
                # Fallback without offset columns
                stmt = (
                    update(Evidence)
                    .where(Evidence.id == candidate.evidence_id)
                    .values(
                        quote=candidate.new_quote,
                        quote_hash=candidate.new_hash,
                    )
                )
                await db.execute(stmt)

            repaired += 1

        await db.commit()

    return repaired


# ============================================================================
# Reporting
# ============================================================================

def print_summary(candidates: list[RepairCandidate], verbose: bool = False) -> None:
    """Print repair candidates summary."""
    if not candidates:
        print("\n  No repair candidates found.")
        return

    similarities = [c.similarity for c in candidates]
    avg_sim = sum(similarities) / len(similarities)
    min_sim = min(similarities)
    max_sim = max(similarities)

    single = sum(1 for c in candidates if c.sentence_count == 1)
    multi = sum(1 for c in candidates if c.sentence_count > 1)

    print(f"\n  Repair candidates:  {len(candidates)}")
    print(f"  Similarity range:   {min_sim:.2%} — {max_sim:.2%}")
    print(f"  Average similarity: {avg_sim:.2%}")
    print(f"  Single-sentence:    {single}")
    print(f"  Multi-sentence:     {multi}")

    if verbose:
        print(f"\n  {'—' * 90}")
        print(f"  {'ID':<38} {'Sim':>5} {'Sents':>5}  Old → New")
        print(f"  {'—' * 90}")
        for c in candidates[:30]:
            old_preview = c.old_quote[:60].replace("\n", " ")
            new_preview = c.new_quote[:60].replace("\n", " ")
            print(f"  {c.evidence_id:<38} {c.similarity:>4.0%}  {c.sentence_count:>4}  "
                  f"\"{old_preview}...\"")
            print(f"  {'':38} {'':>4}  {'':>4}  → \"{new_preview}...\"")
            print()
        if len(candidates) > 30:
            print(f"  ... and {len(candidates) - 30} more")


def export_csv(candidates: list[RepairCandidate], path: str) -> None:
    """Export repair details to CSV."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "evidence_id", "relation_id", "chunk_id", "similarity",
            "sentence_count", "old_quote", "new_quote",
            "new_quote_start", "new_quote_end",
        ])
        for c in candidates:
            writer.writerow([
                c.evidence_id, c.relation_id, c.chunk_id,
                f"{c.similarity:.4f}", c.sentence_count,
                c.old_quote.replace("\n", " "),
                c.new_quote.replace("\n", " "),
                c.new_quote_start, c.new_quote_end,
            ])


# ============================================================================
# Main
# ============================================================================

async def run(
    execute: bool = False,
    threshold: float = DEFAULT_FUZZY_THRESHOLD,
    export_path: str | None = None,
    verbose: bool = False,
) -> dict:
    """Find and optionally repair fuzzy-match evidence quotes."""
    engine = create_async_engine(settings.db_url, echo=False)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    try:
        candidates = await find_repair_candidates(session_factory, threshold)

        print("\n" + "=" * 60)
        print("Evidence Repair Report (Sentence-Aligned)")
        print("=" * 60)
        print(f"\n  Mode:      {'EXECUTE' if execute else 'DRY RUN'}")
        print(f"  Threshold: {threshold:.0%}")

        print_summary(candidates, verbose=verbose)

        if candidates and execute:
            print(f"\n  Applying {len(candidates)} sentence-aligned repairs...")
            repaired = await apply_repairs(session_factory, candidates)
            print(f"  Successfully repaired: {repaired} evidence records")
            print(f"\n  Tip: Re-run audit_evidence.py to verify repairs.")
        elif candidates and not execute:
            print(f"\n  Dry run: {len(candidates)} records would be repaired.")
            print("  Run with --execute to apply changes.")

        if export_path and candidates:
            export_csv(candidates, export_path)
            print(f"\n  Report exported to: {export_path}")

        print()

        return {
            "candidates": len(candidates),
            "repaired": len(candidates) if execute else 0,
            "threshold": threshold,
        }

    finally:
        await engine.dispose()


def main():
    parser = argparse.ArgumentParser(
        description="Repair fuzzy-match evidence quotes (sentence-aligned)",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually apply repairs (default is dry run)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_FUZZY_THRESHOLD,
        help=f"Minimum similarity for repair (default: {DEFAULT_FUZZY_THRESHOLD})",
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="Export repair details to CSV",
    )
    parser.add_argument(
        "--export-path",
        type=str,
        default="demo_output/evidence_repairs.csv",
        help="Path for CSV export (default: demo_output/evidence_repairs.csv)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show individual repair details",
    )
    args = parser.parse_args()

    export_path = args.export_path if args.export else None

    asyncio.run(run(
        execute=args.execute,
        threshold=args.threshold,
        export_path=export_path,
        verbose=args.verbose,
    ))


if __name__ == "__main__":
    main()