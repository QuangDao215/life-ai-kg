"""
Deduplicate evidence records and add unique constraint.

Removes duplicate evidence where the same quote (by hash) exists
for the same relation across different chunks (from chunk overlap).
Keeps the earliest record per (relation_id, quote_hash).

Usage:
    python scripts/dedup_evidence.py              # Dry run (show what would be deleted)
    python scripts/dedup_evidence.py --execute     # Actually delete duplicates
    python scripts/dedup_evidence.py --execute --add-constraint  # Delete + add unique constraint
"""

import argparse
import asyncio

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from src.core.config import settings
from src.core.logging import get_logger

logger = get_logger(__name__)

COUNT_DUPLICATES_SQL = """
SELECT COUNT(*) AS duplicate_count FROM (
    SELECT id,
           ROW_NUMBER() OVER (
               PARTITION BY relation_id, quote_hash
               ORDER BY created_at ASC
           ) AS rn
    FROM evidence
) ranked
WHERE rn > 1;
"""

DELETE_DUPLICATES_SQL = """
DELETE FROM evidence
WHERE id IN (
    SELECT id FROM (
        SELECT id,
               ROW_NUMBER() OVER (
                   PARTITION BY relation_id, quote_hash
                   ORDER BY created_at ASC
               ) AS rn
        FROM evidence
    ) ranked
    WHERE rn > 1
);
"""

ADD_UNIQUE_CONSTRAINT_SQL = """
ALTER TABLE evidence
ADD CONSTRAINT uq_evidence_relation_quote UNIQUE (relation_id, quote_hash);
"""

CHECK_CONSTRAINT_SQL = """
SELECT COUNT(*) FROM information_schema.table_constraints
WHERE table_name = 'evidence'
  AND constraint_name = 'uq_evidence_relation_quote';
"""


async def run(execute: bool = False, add_constraint: bool = False) -> None:
    engine = create_async_engine(settings.db_url, echo=False)

    try:
        async with engine.begin() as conn:
            # Count duplicates
            result = await conn.execute(text(COUNT_DUPLICATES_SQL))
            count = result.scalar()
            print(f"\nDuplicate evidence records found: {count}")

            if count == 0:
                print("No duplicates to remove.")
            elif not execute:
                print(f"Dry run: {count} records would be deleted.")
                print("Run with --execute to delete them.")
            else:
                # Delete duplicates
                result = await conn.execute(text(DELETE_DUPLICATES_SQL))
                print(f"Deleted {result.rowcount} duplicate evidence records.")

            # Handle unique constraint
            if add_constraint:
                # Check if constraint already exists
                result = await conn.execute(text(CHECK_CONSTRAINT_SQL))
                exists = result.scalar() > 0

                if exists:
                    print("Unique constraint uq_evidence_relation_quote already exists.")
                elif not execute:
                    print("Dry run: would add UNIQUE(relation_id, quote_hash) constraint.")
                else:
                    await conn.execute(text(ADD_UNIQUE_CONSTRAINT_SQL))
                    print("Added unique constraint: uq_evidence_relation_quote(relation_id, quote_hash)")

            # Show final count
            result = await conn.execute(text("SELECT COUNT(*) FROM evidence;"))
            total = result.scalar()
            print(f"\nTotal evidence records: {total}")

    finally:
        await engine.dispose()


def main():
    parser = argparse.ArgumentParser(description="Deduplicate evidence records")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually delete duplicates (default is dry run)",
    )
    parser.add_argument(
        "--add-constraint",
        action="store_true",
        help="Add unique constraint on (relation_id, quote_hash)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Evidence Deduplication")
    print("=" * 60)
    if not args.execute:
        print("Mode: DRY RUN (use --execute to apply changes)")
    else:
        print("Mode: EXECUTE")

    asyncio.run(run(execute=args.execute, add_constraint=args.add_constraint))


if __name__ == "__main__":
    main()