# Changelog

All notable changes to the LIFE AI Knowledge Graph project are documented in this file.

---

## [0.3.0] - 2026-02-15

### Summary

Comprehensive evidence quality audit, repair, and root cause diagnosis. Full scan of 2,546
evidence records revealed 92.1% exact-match rate. Identified ellipsis condensation (78%) as
the dominant cause of non-verbatim quotes. Repaired 5 fuzzy-match records using sentence-aligned
matching, diagnosed all 195 non-exact records by root cause, and hardened the extraction prompt
to prevent future occurrences.

### Added

- **`scripts/audit_evidence.py`** — Full evidence validation audit script. Scans all evidence
  records against their source chunks using exact match, case-insensitive match, and fuzzy
  sliding-window matching. Produces summary statistics and optional CSV export with per-record
  details. Supports configurable similarity thresholds and verbose output.

- **`scripts/repair_evidence.py`** — Sentence-aligned evidence repair script. Replaces fuzzy-match
  quotes with complete sentence(s) from the source chunk instead of arbitrary character-level
  substrings. Splits chunks into sentences using biomedical-aware tokenization (handles
  abbreviations like "e.g.", "et al.", "vs.", "fig."). Includes dry-run mode, CSV export, and
  configurable thresholds.

- **`scripts/diagnose_evidence.py`** — Root cause diagnosis script for non-exact evidence records.
  Classifies each record into one of five categories: ELLIPSIS (LLM used "..." to condense),
  CHUNK_EDGE (sentence cut by chunk boundary), PARAPHRASE (LLM rephrased), CROSS_SENT (LLM
  merged multiple sentences), or HALLUCINATED (no recoverable match). Includes confidence
  scoring, secondary cause detection, and actionable recommendations.

### Fixed

- **5 fuzzy-match evidence records repaired** — Sentence-aligned matching replaced truncated
  character-level quotes with clean, complete sentences from the source text. All 5 repairs
  verified at 83–97% similarity with single-sentence alignment.

### Changed

- **Relation extraction prompt hardened against ellipsis** — Added explicit anti-ellipsis and
  verbatim-copy instructions to `RELATION_EXTRACTION_SYSTEM_PROMPT` in `extraction.py`:
  - Evidence field description now states: "Must be a complete, contiguous passage — do NOT
    use '...' or ellipsis to skip content"
  - Rules section expanded with 5 specific verbatim-copy constraints including: no ellipsis,
    no paraphrasing, no reordering, single contiguous passage only, and guidance to select
    shorter complete sentences when text is too long
  - This addresses the dominant root cause (78% of non-exact evidence was caused by LLM
    ellipsis condensation)

### Evidence Quality Metrics

| Metric                  | Before (v0.2.0) | After (v0.3.0) |
|-------------------------|------------------|-----------------|
| Evidence records        | 2,546            | 2,546           |
| Exact match             | ~90% (sampled)   | 92.3% (full audit) |
| Fuzzy match (repaired)  | unknown          | 5 → 0 remaining |
| Non-exact               | unknown          | 195 (diagnosed) |
| Root cause: ELLIPSIS    | —                | 152 (77.9%)     |
| Root cause: CHUNK_EDGE  | —                | 12 (6.2%)       |
| Root cause: HALLUCINATED| —                | 31 (15.9%)      |

---

## [0.2.0] - 2026-02-14

### Summary

Resolved all critical extraction pipeline failures, achieving 99.55% chunk coverage (222/223)
with 0% error rate. The knowledge graph now contains 1,258 entities, 2,174 relations, and
2,546 evidence records across 90 PubMed documents. Evaluation score upgraded from WARN to PASS.

### Fixed

- **Truncated LLM responses causing 96% failure rate** — Gemini relation extraction responses
  were exceeding `max_output_tokens`, producing incomplete JSON that could not be parsed.
  Increased token limit from 2048 to 8192 in the LLM client generation config. This was the
  root cause behind the `Expecting ',' delimiter` JSON parse errors on nearly every chunk.

- **ExtractionResult reporting bug** — `entity_count` and `relation_count` fields were computed
  in `__post_init__` at construction time when entity/relation lists were still empty.
  Replaced with `@property` accessors that dynamically return `len(self.entities)` and
  `len(self.relations)`. Removed stale `__post_init__` and the old `int = 0` field defaults.

- **Duplicate evidence records from chunk overlap** — 79 duplicate evidence records existed
  because the same quote could appear in two adjacent overlapping chunks. Changed the
  deduplication check in `_create_evidence` and `_create_evidence_from_data` to use
  `(relation_id, quote_hash)` instead of `(relation_id, chunk_id, quote_hash)`, preventing
  cross-chunk duplicates. Added `dedup_evidence.py` script for one-time cleanup and optional
  unique constraint.

### Added

- **Truncated JSON repair** — `_repair_json()` in `extraction.py` now detects incomplete JSON
  from truncated API responses and salvages all complete objects by finding the last valid `},`
  and closing the array/object structure. This acts as a safety net if token limits are hit.

- **`scripts/dedup_evidence.py`** — Maintenance script for evidence deduplication with dry-run
  mode, execute mode, and optional `--add-constraint` flag to add a
  `UNIQUE(relation_id, quote_hash)` database constraint.

- **Raw response debug logging** — `_parse_relation_response()` now logs the first 2000
  characters of the raw LLM response on parse failure, enabling rapid diagnosis of malformed
  JSON issues.

### Changed

- **Chunk delay** increased from 2s to 10s between chunks during extraction to reduce
  Gemini free-tier rate limit pressure.

- **Call delay** of 1s added between entity and relation API calls within each chunk.

### Pipeline Statistics

| Metric               | v0.1.0 (initial) | v0.2.0 (current) |
|----------------------|-------------------|-------------------|
| Chunks processed     | 33/223 (14.8%)    | 222/223 (99.55%)  |
| Entities             | 187               | 1,258             |
| Relations            | 215               | 2,174             |
| Evidence records     | ~250              | 2,546             |
| Extraction error rate| 100% (run 3)      | 0%                |
| Duplicate evidence   | 79                | 0                 |
| Evaluation score     | WARN              | PASS              |
| Sanity checks        | 7/7               | 7/7               |

---

## [0.1.0] - 2026-02-12

### Added

- Initial project implementation with full end-to-end pipeline.
- PubMed E-utilities integration for article fetching (esearch + efetch).
- Configurable document chunking (size: 1000 chars, overlap: 200 chars).
- LLM-based entity and relation extraction using Google Gemini 2.5 Flash.
- 9 entity types: Disease, Gene, Protein, Pathway, CellType, Tissue/Region,
  Compound/Drug, Phenotype, Biomarker.
- 11 relation labels: ASSOCIATED_WITH, INCREASES_RISK, DECREASES_RISK, UPREGULATES,
  DOWNREGULATES, ACTIVATES, INHIBITS, BINDS, BIOMARKER_FOR, PART_OF, EXPRESSED_IN.
- Evidence linking with exact quote offsets for provenance tracking.
- Async job pipeline via Celery + Redis with progress tracking.
- REST API endpoints: entity search, neighborhood view, bounded subgraph, path computation.
- Idempotent upserts for entities, relations, and evidence.
- Evaluation harness with summary statistics, quality checks, and sanity queries.
- Interactive knowledge graph visualization (D3.js-based HTML).
- End-to-end demo script (`run-demo.ps1`).
- PostgreSQL + Redis via Docker Compose.