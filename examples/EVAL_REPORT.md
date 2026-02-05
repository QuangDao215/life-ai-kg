# 📊 LIFE AI Knowledge Graph — Evaluation Report

> Generated: 2026-02-05T14:35:26.838178

## Overall Score: ⚠️ WARN

---

## Summary Statistics

| Metric | Value |
|--------|-------|
| Documents | 20 |
| Chunks | 53 |
| Chunks Processed | 11 (20.8%) |
| Entities | 44 |
| Relations | 57 |
| Evidence Records | 68 |
| Evidence Coverage | 100.0% |
| Avg Confidence | 0.89 |
| Confidence Range | 0.70 – 0.95 |

### Entity Types

| Type | Count |
|------|-------|
| Biomarker | 10 |
| Phenotype | 7 |
| Disease | 7 |
| Pathway | 5 |
| Tissue/Region | 5 |
| Gene | 4 |
| Protein | 4 |
| Compound/Drug | 2 |

### Relation Labels

| Label | Count |
|-------|-------|
| ASSOCIATED_WITH | 20 |
| INCREASES_RISK | 9 |
| EXPRESSED_IN | 8 |
| UPREGULATES | 7 |
| PART_OF | 4 |
| DOWNREGULATES | 3 |
| BIOMARKER_FOR | 2 |
| DECREASES_RISK | 2 |
| BINDS | 2 |

---

## Quality Metrics

| Check | Value |
|-------|-------|
| Duplicate Entities | 0 |
| Duplicate Relations | 0 |
| Duplicate Evidence | 2 |
| Orphan Relations | 0 |
| Relations Without Evidence | 0 |
| Invalid Entity Types | 0 |
| Invalid Relation Labels | 0 |
| Entities Without Relations | 2 |
| Avg Relations per Entity | 2.6 |
| Avg Evidence per Relation | 1.2 |

### ⚠️ Issues Found

- Found 2 duplicate evidence records

---

## Sanity Checks

**7 / 7 passed**

| # | Check | Result | Details |
|---|-------|--------|---------|
| 1 | ambroxol_exists | ✅ | Found 1 Ambroxol entities with 16 relations |
| 2 | parkinsons_exists | ✅ | Found 4 Parkinson's entities with 14 relations |
| 3 | mechanistic_entities | ✅ | Found 4/5 key mechanistic terms |
| 4 | entity_type_diversity | ✅ | Found 8 entity types: ['Protein', 'Disease', 'Tissue/Region', 'Phenotype', 'Pathway', 'Compound/Drug', 'Gene', 'Biomarker'] |
| 5 | relation_label_diversity | ✅ | Found 9 relation types: ['PART_OF', 'EXPRESSED_IN', 'UPREGULATES', 'ASSOCIATED_WITH', 'DOWNREGULATES', 'DECREASES_RISK', 'BIOMARKER_FOR', 'INCREASES_RISK', 'BINDS'] |
| 6 | evidence_validity | ✅ | Sampled 10 evidence: 9 valid, 1 invalid |
| 7 | ambroxol_pd_connection | ✅ | Direct connection exists between Ambroxol and Parkinson's |

---

## Example Queries

### Query

### Query

### Query
