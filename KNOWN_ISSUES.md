# Known Issues & Areas for Improvement

**Last updated:** February 15, 2026
**Pipeline state:** 222/223 chunks processed (99.55%), 1,258 entities, 2,174 relations
**Evaluation score:** PASS (7/7 sanity checks)

---

## Resolved Issues

### ~~1. Extraction Errors on Every Chunk~~ ✅ RESOLVED

**Severity:** High → **Fixed in v0.2.0**

**Original problem:** Runs 2 and 3 had 84–96% failure rates. Every chunk failed on relation
extraction with `Expecting ',' delimiter` JSON parse errors. Initially misdiagnosed as
Gemini API rate limiting.

**Root cause:** `max_output_tokens` was set too low (2048). Gemini's relation responses were
being truncated mid-JSON, producing incomplete output that no parser could recover.

**Resolution:**
- Increased `max_output_tokens` to 8192 in the LLM client generation config.
- Added truncated JSON repair in `_repair_json()` as a safety net — finds the last complete
  JSON object and closes the array/object structure.
- Added raw response logging (first 2000 chars) to `_parse_relation_response()` for faster
  diagnosis of future parse failures.
- **Result:** 0% error rate across 97 consecutive chunks.

---

### ~~2. Duplicate Evidence Records~~ ✅ RESOLVED

**Severity:** Low → **Fixed in v0.2.0**

**Original problem:** 79 duplicate evidence records found. Same quote appeared in two adjacent
chunks due to 200-character overlap, creating separate evidence rows for the same relation.

**Root cause:** The deduplication check used `(relation_id, chunk_id, quote_hash)`, which
allowed duplicates across different chunks.

**Resolution:**
- Changed dedup check to `(relation_id, quote_hash)` — ignoring `chunk_id` — in both
  `_create_evidence()` and `_create_evidence_from_data()`.
- Created `scripts/dedup_evidence.py` for one-time cleanup of existing duplicates.
- Optional `UNIQUE(relation_id, quote_hash)` constraint available via `--add-constraint` flag.
- **Result:** 0 duplicate evidence records. Evidence count reduced from 2,625 to 2,546.

---

### ~~3. ExtractionResult Reporting Bug~~ ✅ RESOLVED

**Severity:** Medium → **Fixed in v0.2.0**

**Original problem:** Batch extraction summary always showed `total_entities: 0` and
`total_relations: 0` despite successful extraction (visible in database counts).

**Root cause:** `ExtractionResult` dataclass computed `entity_count` and `relation_count` in
`__post_init__` at construction time, when the entity/relation lists were still empty. The
lists were populated after construction, but the counts were never recalculated.

**Resolution:**
- Replaced `entity_count: int = 0` and `relation_count: int = 0` fields with `@property`
  accessors that dynamically return `len(self.entities)` and `len(self.relations)`.
- Removed `__post_init__` entirely.
- **Result:** CLI extraction summary now reports accurate per-run counts. Celery job results
  show correct `total_entities` and `total_relations` when processing new chunks.

---

### ~~4. Low Corpus Coverage~~ ✅ RESOLVED

**Severity:** Medium → **Fixed in v0.2.0**

**Original problem:** Only 33/223 chunks (14.8%) had been processed. Multiple extraction runs
failed due to rate limits and the JSON truncation bug.

**Resolution:**
- Fixed the truncation bug (Issue #1) which was the primary blocker.
- Increased `chunk_delay_seconds` from 2 to 10 to stay within Gemini free-tier rate limits.
- Ran extraction in batches across multiple sessions (quota resets daily).
- **Result:** 222/223 chunks processed (99.55%). The 1 remaining chunk is from a document
  with an empty abstract.

---

## Open Issues

### 5. Evidence Validity (Non-Verbatim Quotes) — PARTIALLY RESOLVED

**Severity:** Medium → **Investigated and partially fixed in v0.3.0**

**Original estimate:** ~10% hallucination rate (1/10 sampled).

**Full audit results (2,546 records):**
- Exact match: 2,346 (92.1%) ✅
- Fuzzy match (repairable): 5 (0.2%) → repaired via sentence-aligned matching ✅
- Non-exact (unrepairable): 195 (7.7%)

**Root cause diagnosis (195 non-exact records):**

| Cause | Count | % | Description |
|-------|-------|---|-------------|
| ELLIPSIS | 152 | 77.9% | LLM used "..." to condense/skip content in quotes |
| CHUNK_EDGE | 12 | 6.2% | Quote overlaps chunk boundary (sentence cut by chunking) |
| HALLUCINATED | 31 | 15.9% | No recoverable match in source chunk |
| PARAPHRASE | 0 | 0% | — |
| CROSS_SENT | 0 | 0% | — |

**Resolution applied:**
- [x] Full evidence audit script (`scripts/audit_evidence.py`) — scans all 2,546 records
- [x] Sentence-aligned repair script (`scripts/repair_evidence.py`) — repaired 5 fuzzy matches
- [x] Root cause diagnosis script (`scripts/diagnose_evidence.py`) — classified all 195 failures
- [x] Prompt fix in `RELATION_EXTRACTION_SYSTEM_PROMPT` — added explicit anti-ellipsis and
  verbatim copy instructions to prevent the dominant failure mode (78% of issues)

**Remaining work:**
- [ ] Remove or flag 31 hallucinated evidence records (no recoverable match)
- [ ] Re-extract affected chunks with updated prompt to regenerate clean evidence
- [ ] Consider sentence-aware chunking to eliminate chunk-edge issues (6%)
- [ ] Re-run full audit after next extraction to measure improvement

---

### 6. Empty Abstract Handling

**Severity:** Low
**Observation:** 4 documents have `abstract = ""` (empty string) instead of `NULL`, causing
them to appear in "Needing chunking" statistics but producing 0 chunks. This also leaves
1 chunk perpetually unprocessed in coverage stats.

**Possible improvements:**
- [ ] Fix PubMed ingestion to store `NULL` instead of empty string when no abstract exists
- [ ] One-time fix: `UPDATE documents SET abstract = NULL WHERE abstract = ''`
- [ ] Update chunking query to exclude empty strings: `WHERE abstract IS NOT NULL AND abstract != ''`

---

### 7. Orphan Entities (No Relations)

**Severity:** Low
**Observation:** 119 entities (9.5%) have no relations. These were extracted by the entity
pass but never linked by the relation pass.

**Context:** This is partially expected in biomedical NER — some entities are legitimate
standalone mentions (background context, method descriptions). The orphan rate decreased
from 11.2% to 9.5% as more chunks were processed, suggesting many were resolved by
additional context from other chunks.

**Possible improvements:**
- [ ] Review orphan entities to classify as quality issue vs. genuinely standalone
- [ ] Consider post-processing step to prune low-value orphan entities
- [ ] Tune entity extraction prompt to be more selective

---

### 8. SQLAlchemy Logging Noise

**Severity:** Low
**Observation:** When `echo=True` is set on the engine, SQLAlchemy prints every SQL statement
with full parameter lists (500+ UUIDs), completely burying useful extraction logs.

**Workaround:** Set `echo=False` in engine creation or add to extraction scripts:
```python
import logging
logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
```

**Status:** Fixed in Celery task engine (`echo=False`). CLI scripts may still need the
logging suppression depending on configuration.

---

## Quality Metrics Summary

| Metric                    | Value           |
|---------------------------|-----------------|
| Documents                 | 90              |
| Chunks                    | 223             |
| Chunks processed          | 222 (99.55%)    |
| Entities                  | 1,258           |
| Relations                 | 2,174           |
| Evidence records          | 2,546           |
| Evidence exact match      | 2,346 (92.1%)   |
| Evidence repaired         | 5 (0.2%)        |
| Evidence non-exact        | 195 (7.7%)      |
| Evidence coverage         | 100%            |
| Avg confidence            | 0.897           |
| Duplicate entities        | 0               |
| Duplicate relations       | 0               |
| Duplicate evidence        | 0               |
| Orphan entities           | 119 (9.5%)      |
| Relations without evidence| 0               |
| Avg relations per entity  | 3.46            |
| Avg evidence per relation | 1.17            |
| Entity types              | 9/9             |
| Relation labels           | 11/11           |
| Sanity checks             | 7/7 PASS        |