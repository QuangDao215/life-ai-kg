"""
Diagnose root causes of non-exact evidence quotes.

Classifies each non-exact evidence record into probable root cause:

1. ELLIPSIS    — LLM used "..." to condense/skip content in the quote
2. CHUNK_EDGE  — Quote overlaps with chunk boundaries (first/last N chars),
                 suggesting the source sentence was cut by chunking
3. PARAPHRASE  — A single sentence in the chunk has high similarity (>70%),
                 but the LLM rephrased it (word order, synonyms, articles)
4. CROSS_SENT  — Quote content spans multiple sentences (LLM merged them)
5. HALLUCINATED — No clear match to any sentence; likely fabricated

Also reports chunk-boundary statistics to quantify the chunking problem.

Usage:
    python scripts/diagnose_evidence.py                # Full diagnosis
    python scripts/diagnose_evidence.py --export       # Save CSV report
    python scripts/diagnose_evidence.py --show-all     # Print every record's diagnosis
    python scripts/diagnose_evidence.py --limit 50     # Diagnose first 50 non-exact
"""

import argparse
import asyncio
import csv
import os
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from src.core.config import settings
from src.core.logging import get_logger
from src.db.models import Evidence, Chunk

logger = get_logger(__name__)


# ============================================================================
# Configuration
# ============================================================================

# How many chars from chunk start/end to consider "edge zone"
CHUNK_EDGE_ZONE = 200

# Minimum sentence-level similarity to count as paraphrase
PARAPHRASE_THRESHOLD = 0.70

# Minimum combined similarity across 2+ sentences for cross-sentence
CROSS_SENTENCE_THRESHOLD = 0.60

# Ellipsis patterns
ELLIPSIS_RE = re.compile(r'\.{2,}|…')

# Sentence split regex (same as repair script)
SENTENCE_SPLIT_RE = re.compile(
    r'(?<=[.!?])'
    r'(?:\s+)'
    r'(?=[A-Z([])'
)

ABBREVIATIONS = {
    "e.g", "i.e", "et al", "vs", "fig", "figs", "ref", "refs",
    "vol", "no", "approx", "ca", "dept", "dr", "mr", "mrs", "ms",
    "prof", "inc", "ltd", "co", "corp", "jr", "sr", "st",
}


# ============================================================================
# Data structures
# ============================================================================

CAUSES = [
    "ELLIPSIS",
    "CHUNK_EDGE",
    "PARAPHRASE",
    "CROSS_SENT",
    "HALLUCINATED",
]


@dataclass
class Diagnosis:
    """Diagnosis for a single non-exact evidence record."""
    evidence_id: str
    chunk_id: str
    relation_id: str
    cause: str                  # primary cause from CAUSES
    secondary_causes: list[str] = field(default_factory=list)
    confidence: str = ""        # HIGH / MEDIUM / LOW

    quote_preview: str = ""
    best_sentence: str = ""     # best matching sentence in chunk
    best_sentence_sim: float = 0.0

    has_ellipsis: bool = False
    is_chunk_edge: bool = False
    edge_zone: str = ""         # "start", "end", "both", ""
    chunk_position: str = ""    # character position info

    # For cross-sentence detection
    top2_sentences_sim: float = 0.0


@dataclass
class SentenceSpan:
    text: str
    start: int
    end: int


# ============================================================================
# Sentence splitting
# ============================================================================

def split_sentences(text: str) -> list[SentenceSpan]:
    """Split text into sentences, preserving offsets."""
    if not text.strip():
        return []

    splits = list(SENTENCE_SPLIT_RE.finditer(text))

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

    sentences: list[SentenceSpan] = []
    prev_end = 0

    for match in valid_splits:
        sent_end = match.start()
        next_start = match.end()
        sent_text = text[prev_end:sent_end].strip()
        if sent_text:
            sentences.append(SentenceSpan(text=sent_text, start=prev_end, end=sent_end))
        prev_end = next_start

    remaining = text[prev_end:].strip()
    if remaining:
        sentences.append(SentenceSpan(text=remaining, start=prev_end, end=len(text)))

    if not sentences:
        sentences.append(SentenceSpan(text=text.strip(), start=0, end=len(text)))

    return sentences


# ============================================================================
# Diagnostic checks
# ============================================================================

def check_ellipsis(quote: str) -> bool:
    """Does the quote contain '...' or '…' (LLM condensation)?"""
    return bool(ELLIPSIS_RE.search(quote))


def check_chunk_edge(
    quote: str,
    chunk_text: str,
    edge_zone: int = CHUNK_EDGE_ZONE,
) -> tuple[bool, str]:
    """
    Does the quote content primarily match the first or last N chars of chunk?

    Returns (is_edge, zone) where zone is "start", "end", "both", or "".
    """
    quote_lower = quote.lower()
    chunk_lower = chunk_text.lower()
    chunk_len = len(chunk_lower)

    if chunk_len < edge_zone * 2:
        # Chunk is too small for meaningful edge detection
        return False, ""

    start_zone = chunk_lower[:edge_zone]
    end_zone = chunk_lower[-edge_zone:]
    middle_zone = chunk_lower[edge_zone:-edge_zone]

    # Check which zone has highest similarity to the quote
    start_sim = SequenceMatcher(None, quote_lower, start_zone).ratio()
    end_sim = SequenceMatcher(None, quote_lower, end_zone).ratio()
    middle_sim = SequenceMatcher(None, quote_lower, middle_zone).ratio() if middle_zone else 0.0

    # It's a chunk edge issue if edge similarity is notably higher than middle
    is_start = start_sim > middle_sim and start_sim > 0.4
    is_end = end_sim > middle_sim and end_sim > 0.4

    if is_start and is_end:
        return True, "both"
    elif is_start:
        return True, "start"
    elif is_end:
        return True, "end"
    return False, ""


def check_paraphrase(
    quote: str,
    sentences: list[SentenceSpan],
) -> tuple[bool, str, float]:
    """
    Is there a single sentence with high similarity to the quote?

    Returns (is_paraphrase, best_sentence_text, similarity).
    """
    best_sim = 0.0
    best_sent = ""

    for sent in sentences:
        sim = SequenceMatcher(None, quote.lower(), sent.text.lower()).ratio()
        if sim > best_sim:
            best_sim = sim
            best_sent = sent.text

    return best_sim >= PARAPHRASE_THRESHOLD, best_sent, best_sim


def check_cross_sentence(
    quote: str,
    sentences: list[SentenceSpan],
) -> tuple[bool, float]:
    """
    Does the quote content span multiple sentences?

    Checks if combining the top-2 matching sentences produces a better
    similarity than any single sentence alone.
    """
    if len(sentences) < 2:
        return False, 0.0

    # Score each sentence against the quote
    scored = []
    for i, sent in enumerate(sentences):
        sim = SequenceMatcher(None, quote.lower(), sent.text.lower()).ratio()
        scored.append((sim, i))

    scored.sort(reverse=True)

    # Try combining top-2 matching sentences (if they're adjacent)
    best_combined = 0.0
    for a in range(min(3, len(scored))):
        for b in range(a + 1, min(4, len(scored))):
            idx_a = scored[a][1]
            idx_b = scored[b][1]

            # Only combine if adjacent or near-adjacent
            if abs(idx_a - idx_b) <= 2:
                lo = min(idx_a, idx_b)
                hi = max(idx_a, idx_b)
                combined = " ".join(sentences[j].text for j in range(lo, hi + 1))
                sim = SequenceMatcher(None, quote.lower(), combined.lower()).ratio()
                if sim > best_combined:
                    best_combined = sim

    # It's cross-sentence if the combined score beats the best single score
    best_single = scored[0][0] if scored else 0.0
    is_cross = (
        best_combined > best_single + 0.05
        and best_combined >= CROSS_SENTENCE_THRESHOLD
    )

    return is_cross, best_combined


# ============================================================================
# Main diagnosis logic
# ============================================================================

def diagnose_record(
    quote: str,
    chunk_text: str,
    sentences: list[SentenceSpan],
) -> Diagnosis:
    """
    Run all diagnostic checks and determine the primary root cause.

    Priority order (first match wins as primary):
    1. ELLIPSIS — definitive signal
    2. CHUNK_EDGE — structural cause
    3. PARAPHRASE — LLM behavioral cause
    4. CROSS_SENT — LLM behavioral cause
    5. HALLUCINATED — fallback
    """
    diag = Diagnosis(
        evidence_id="",
        chunk_id="",
        relation_id="",
        cause="HALLUCINATED",  # default
        quote_preview=quote[:80].replace("\n", " "),
    )

    # Check 1: Ellipsis
    diag.has_ellipsis = check_ellipsis(quote)

    # Check 2: Chunk edge
    diag.is_chunk_edge, diag.edge_zone = check_chunk_edge(quote, chunk_text)

    # Check 3: Paraphrase
    is_paraphrase, best_sent, best_sim = check_paraphrase(quote, sentences)
    diag.best_sentence = best_sent[:80].replace("\n", " ")
    diag.best_sentence_sim = best_sim

    # Check 4: Cross-sentence
    is_cross, cross_sim = check_cross_sentence(quote, sentences)
    diag.top2_sentences_sim = cross_sim

    # Determine primary cause (priority order)
    secondary = []

    if diag.has_ellipsis:
        diag.cause = "ELLIPSIS"
        diag.confidence = "HIGH"
        if diag.is_chunk_edge:
            secondary.append("CHUNK_EDGE")
        if is_paraphrase:
            secondary.append("PARAPHRASE")
    elif diag.is_chunk_edge:
        diag.cause = "CHUNK_EDGE"
        diag.confidence = "HIGH" if diag.edge_zone in ("start", "end") else "MEDIUM"
        if is_paraphrase:
            secondary.append("PARAPHRASE")
    elif is_paraphrase:
        diag.cause = "PARAPHRASE"
        diag.confidence = "HIGH" if best_sim >= 0.80 else "MEDIUM"
        if is_cross:
            secondary.append("CROSS_SENT")
    elif is_cross:
        diag.cause = "CROSS_SENT"
        diag.confidence = "MEDIUM"
    else:
        diag.cause = "HALLUCINATED"
        diag.confidence = "LOW" if best_sim >= 0.50 else "HIGH"

    diag.secondary_causes = secondary
    return diag


# ============================================================================
# Database scan
# ============================================================================

async def run_diagnosis(
    limit: int | None = None,
    export_path: str | None = None,
    show_all: bool = False,
) -> dict:
    """Scan non-exact evidence records and diagnose root causes."""

    engine = create_async_engine(settings.db_url, echo=False)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    try:
        async with session_factory() as db:
            total_result = await db.execute(select(func.count(Evidence.id)))
            total_evidence = total_result.scalar()

            query = (
                select(Evidence, Chunk.text)
                .join(Chunk, Evidence.chunk_id == Chunk.id)
                .order_by(Evidence.created_at)
            )
            result = await db.execute(query)
            rows = result.all()

        # Filter to non-exact records only
        non_exact = []
        for evidence, chunk_text in rows:
            if evidence.quote not in chunk_text:
                if evidence.quote.lower() not in chunk_text.lower():
                    non_exact.append((evidence, chunk_text))

        if limit:
            non_exact = non_exact[:limit]

        print(f"\nDiagnosing {len(non_exact)} non-exact evidence records...\n")

        diagnoses: list[Diagnosis] = []
        counts = {c: 0 for c in CAUSES}

        for i, (evidence, chunk_text) in enumerate(non_exact):
            sentences = split_sentences(chunk_text)
            diag = diagnose_record(evidence.quote, chunk_text, sentences)
            diag.evidence_id = str(evidence.id)
            diag.chunk_id = str(evidence.chunk_id)
            diag.relation_id = str(evidence.relation_id)
            diagnoses.append(diag)
            counts[diag.cause] += 1

            if (i + 1) % 50 == 0 or (i + 1) == len(non_exact):
                print(f"  Diagnosed {i + 1}/{len(non_exact)}...", flush=True)

        diagnosed = len(diagnoses)

        # ====== Print report ======
        print("\n" + "=" * 60)
        print("Evidence Diagnosis Report")
        print("=" * 60)

        print(f"\n  Total evidence in database: {total_evidence}")
        print(f"  Non-exact records:          {diagnosed}")

        print(f"\n  Root Cause Breakdown:")
        print(f"  {'—' * 50}")
        for cause in CAUSES:
            pct = f"{counts[cause] / diagnosed * 100:.1f}%" if diagnosed > 0 else "0%"
            bar = "█" * int(counts[cause] / max(diagnosed, 1) * 30)
            print(f"    {cause:<15} {counts[cause]:>5}  ({pct:>5})  {bar}")

        # Secondary causes
        sec_counts = {}
        for d in diagnoses:
            for sc in d.secondary_causes:
                sec_counts[sc] = sec_counts.get(sc, 0) + 1

        if sec_counts:
            print(f"\n  Secondary Causes (co-occurring):")
            for sc, cnt in sorted(sec_counts.items(), key=lambda x: -x[1]):
                print(f"    {sc:<15} {cnt:>5}")

        # Chunk edge details
        edge_diags = [d for d in diagnoses if d.is_chunk_edge]
        if edge_diags:
            start_count = sum(1 for d in edge_diags if d.edge_zone == "start")
            end_count = sum(1 for d in edge_diags if d.edge_zone == "end")
            both_count = sum(1 for d in edge_diags if d.edge_zone == "both")
            print(f"\n  Chunk Edge Details (among all records with edge signal):")
            print(f"    Edge zone 'start': {start_count}")
            print(f"    Edge zone 'end':   {end_count}")
            print(f"    Edge zone 'both':  {both_count}")
            print(f"    Total with edge:   {len(edge_diags)}")

        # Paraphrase similarity distribution
        para_diags = [d for d in diagnoses if d.cause == "PARAPHRASE"]
        if para_diags:
            sims = [d.best_sentence_sim for d in para_diags]
            avg_sim = sum(sims) / len(sims)
            print(f"\n  Paraphrase Similarity Distribution:")
            print(f"    Average best-sentence sim: {avg_sim:.2%}")
            buckets = {
                "90-100%": sum(1 for s in sims if s >= 0.90),
                "80-90%": sum(1 for s in sims if 0.80 <= s < 0.90),
                "70-80%": sum(1 for s in sims if 0.70 <= s < 0.80),
            }
            for label, cnt in buckets.items():
                print(f"    {label}: {cnt}")

        # Confidence summary
        conf_counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
        for d in diagnoses:
            conf_counts[d.confidence] = conf_counts.get(d.confidence, 0) + 1
        print(f"\n  Confidence in Diagnosis:")
        for conf in ("HIGH", "MEDIUM", "LOW"):
            print(f"    {conf:<8} {conf_counts[conf]:>5}")

        # Show individual records if requested
        if show_all:
            print(f"\n{'=' * 60}")
            print("Individual Diagnoses")
            print("=" * 60)
            for d in diagnoses:
                sec = f" + {', '.join(d.secondary_causes)}" if d.secondary_causes else ""
                print(f"\n  ID:        {d.evidence_id}")
                print(f"  Cause:     {d.cause}{sec}  (confidence: {d.confidence})")
                print(f"  Quote:     \"{d.quote_preview}...\"")
                if d.best_sentence:
                    print(f"  Best sent: \"{d.best_sentence}...\" ({d.best_sentence_sim:.0%})")
                if d.has_ellipsis:
                    print(f"  Ellipsis:  Yes")
                if d.is_chunk_edge:
                    print(f"  Edge zone: {d.edge_zone}")

        # Export
        if export_path:
            _export_csv(diagnoses, export_path)
            print(f"\n  Report exported to: {export_path}")

        # Recommendations
        print(f"\n{'=' * 60}")
        print("Recommendations")
        print("=" * 60)

        if counts["ELLIPSIS"] > 0:
            pct = counts["ELLIPSIS"] / diagnosed * 100
            print(f"\n  [{pct:.0f}%] ELLIPSIS — Prompt fix")
            print(f"    Add to extraction prompt: 'Copy quotes EXACTLY from the")
            print(f"    text. Never use ... or ellipsis to skip content.'")

        if counts["CHUNK_EDGE"] > 0:
            pct = counts["CHUNK_EDGE"] / diagnosed * 100
            print(f"\n  [{pct:.0f}%] CHUNK_EDGE — Chunking fix")
            print(f"    Use sentence-aware chunking that splits on sentence")
            print(f"    boundaries instead of fixed character positions.")

        if counts["PARAPHRASE"] > 0:
            pct = counts["PARAPHRASE"] / diagnosed * 100
            print(f"\n  [{pct:.0f}%] PARAPHRASE — Prompt fix")
            print(f"    Strengthen the verbatim instruction: 'The quote field")
            print(f"    must be an EXACT copy-paste from the text. Do not")
            print(f"    rephrase, reorder, or add/remove any words.'")

        if counts["CROSS_SENT"] > 0:
            pct = counts["CROSS_SENT"] / diagnosed * 100
            print(f"\n  [{pct:.0f}%] CROSS_SENT — Prompt fix")
            print(f"    Add: 'Each quote must come from a single contiguous")
            print(f"    passage. Do not merge content from different sentences.'")

        if counts["HALLUCINATED"] > 0:
            pct = counts["HALLUCINATED"] / diagnosed * 100
            print(f"\n  [{pct:.0f}%] HALLUCINATED — Remove or re-extract")
            print(f"    These {counts['HALLUCINATED']} records have no recoverable match.")
            print(f"    Recommend: flag as invalid or delete.")

        print()

        return {
            "total_evidence": total_evidence,
            "diagnosed": diagnosed,
            "counts": counts,
        }

    finally:
        await engine.dispose()


def _export_csv(diagnoses: list[Diagnosis], path: str) -> None:
    """Export diagnoses to CSV."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "evidence_id", "chunk_id", "relation_id",
            "primary_cause", "secondary_causes", "confidence",
            "has_ellipsis", "is_chunk_edge", "edge_zone",
            "best_sentence_sim", "top2_sentences_sim",
            "quote_preview", "best_sentence",
        ])
        for d in diagnoses:
            writer.writerow([
                d.evidence_id, d.chunk_id, d.relation_id,
                d.cause, "|".join(d.secondary_causes), d.confidence,
                d.has_ellipsis, d.is_chunk_edge, d.edge_zone,
                f"{d.best_sentence_sim:.4f}", f"{d.top2_sentences_sim:.4f}",
                d.quote_preview, d.best_sentence,
            ])


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Diagnose root causes of non-exact evidence quotes",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Max non-exact records to diagnose (default: all)",
    )
    parser.add_argument(
        "--export", action="store_true",
        help="Export diagnosis CSV to demo_output/evidence_diagnosis.csv",
    )
    parser.add_argument(
        "--export-path", type=str,
        default="demo_output/evidence_diagnosis.csv",
        help="Path for CSV export",
    )
    parser.add_argument(
        "--show-all", action="store_true",
        help="Print individual diagnosis for each record",
    )
    args = parser.parse_args()

    export_path = args.export_path if args.export else None

    asyncio.run(run_diagnosis(
        limit=args.limit,
        export_path=export_path,
        show_all=args.show_all,
    ))


if __name__ == "__main__":
    main()