# 📊 LIFE AI Knowledge Graph — Evaluation Report

> Generated: 2026-02-14T11:51:02.259640

## Overall Score: ✅ PASS

---

## Summary Statistics

| Metric | Value |
|--------|-------|
| Documents | 90 |
| Chunks | 223 |
| Chunks Processed | 222 (99.5%) |
| Entities | 1258 |
| Relations | 2174 |
| Evidence Records | 2546 |
| Evidence Coverage | 100.0% |
| Avg Confidence | 0.90 |
| Confidence Range | 0.70 – 1.00 |

### Entity Types

| Type | Count |
|------|-------|
| Phenotype | 343 |
| Compound/Drug | 205 |
| Disease | 147 |
| Protein | 141 |
| Biomarker | 121 |
| Pathway | 99 |
| CellType | 89 |
| Tissue/Region | 58 |
| Gene | 55 |

### Relation Labels

| Label | Count |
|-------|-------|
| ASSOCIATED_WITH | 945 |
| PART_OF | 236 |
| EXPRESSED_IN | 194 |
| INCREASES_RISK | 162 |
| UPREGULATES | 148 |
| BIOMARKER_FOR | 142 |
| DECREASES_RISK | 106 |
| DOWNREGULATES | 88 |
| ACTIVATES | 62 |
| INHIBITS | 61 |
| BINDS | 30 |

---

## Quality Metrics

| Check | Value |
|-------|-------|
| Duplicate Entities | 0 |
| Duplicate Relations | 0 |
| Duplicate Evidence | 0 |
| Orphan Relations | 0 |
| Relations Without Evidence | 0 |
| Invalid Entity Types | 0 |
| Invalid Relation Labels | 0 |
| Entities Without Relations | 119 |
| Avg Relations per Entity | 3.5 |
| Avg Evidence per Relation | 1.2 |

---

## Sanity Checks

**7 / 7 passed**

| # | Check | Result | Details |
|---|-------|--------|---------|
| 1 | ambroxol_exists | ✅ | Found 2 Ambroxol entities with 259 relations |
| 2 | parkinsons_exists | ✅ | Found 33 Parkinson's entities with 280 relations |
| 3 | mechanistic_entities | ✅ | Found 4/5 key mechanistic terms |
| 4 | entity_type_diversity | ✅ | Found 9 entity types: ['Phenotype', 'CellType', 'Pathway', 'Compound/Drug', 'Gene', 'Biomarker', 'Protein', 'Disease', 'Tissue/Region'] |
| 5 | relation_label_diversity | ✅ | Found 11 relation types: ['PART_OF', 'EXPRESSED_IN', 'ACTIVATES', 'UPREGULATES', 'ASSOCIATED_WITH', 'DOWNREGULATES', 'DECREASES_RISK', 'BIOMARKER_FOR', 'INCREASES_RISK', 'BINDS', 'INHIBITS'] |
| 6 | evidence_validity | ✅ | Sampled 10 evidence: 9 valid, 1 invalid |
| 7 | ambroxol_pd_connection | ✅ | Direct connection exists between Ambroxol and Parkinson's |

---

## Example Queries

### Query

### Query

### Query
