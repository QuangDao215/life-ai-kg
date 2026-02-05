# DESIGN.md — Technical Design Document

## LIFE AI Knowledge Graph Backend

**Version:** 0.1.0
**Date:** February 2026
**Author:** Rito

---

## 1. Problem Statement

Biomedical researchers studying Ambroxol as a potential therapy for Parkinson's disease face a growing body of literature spanning molecular biology, pharmacology, and clinical trials. Manually tracking how entities (drugs, genes, proteins, pathways, diseases) interconnect across hundreds of publications is infeasible.

This project builds an automated Knowledge Graph (KG) extraction pipeline that ingests PubMed literature, extracts structured biomedical entities and relationships using LLMs, and serves them through a REST API for programmatic exploration.

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                  Client Layer                       │
│  curl / Swagger UI / Visualization (pyvis HTML)     │
└────────────────┬────────────────────────────────────┘
                 │ HTTP
┌────────────────▼────────────────────────────────────┐
│              FastAPI Application                    │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌──────────┐      │
│  │  Docs  │ │  Jobs  │ │Entities│ │  Graph   │      │
│  │  API   │ │  API   │ │  API   │ │   API    │      │
│  └────┬───┘ └───┬────┘ └───┬────┘ └────┬─────┘      │
│       └─────────┴──────────┴────────────┘           │
│              Pydantic Schemas + SQLAlchemy          │
└────────────────┬────────────────────────────────────┘
                 │ asyncpg
┌────────────────▼─────────────────────────────────────┐
│            PostgreSQL 16                             │
│  documents │ chunks │ entities │ relations │ evidence│
└────────────┬─────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────┐
│          Extraction Pipeline                        │
│  PubMed API → Chunking → Gemini LLM → KG Upserts    │
└─────────────────────────────────────────────────────┘
```

The system follows a three-stage pipeline:

1. **Collection** — Fetch abstracts and metadata from PubMed via E-utilities API
2. **Extraction** — Chunk documents and use Google Gemini to extract entities and relations
3. **Serving** — Expose the KG through typed REST endpoints with evidence provenance

### Async Pipeline Architecture

The KG build pipeline uses a **Celery queue/worker architecture** backed by Redis:

```
┌──────────────┐       ┌──────────────┐        ┌──────────────────┐
│  FastAPI API │──────>│  Redis Queue │──────> │  Celery Worker   │
│              │  task │  (broker)    │  pull  │  -P solo         │
│ POST /jobs/  │       └──────────────┘        │                  │
│   kg-build   │                               │  asyncio.run()   │
└──────┬───────┘       ┌───────────────┐       │  → extraction    │
       │               │  Redis Store  │<──────│  → progress      │
       │    read       │  (job state)  │ write │  → upserts       │
       └──────────────>│               │       └──────────────────┘
  GET /jobs/{id}       └───────────────┘
```

**Components:**
- **API process** — Creates jobs in Redis store, dispatches Celery tasks to queue, returns 202 Accepted
- **Redis** — Serves as both Celery message broker and shared job state store
- **Worker process** — Runs `celery -A src.workers.celery_app worker -P solo`, picks tasks from queue, executes async extraction via `asyncio.run()`, writes progress to Redis
- **Fallback** — If Redis/Celery is unavailable, falls back to FastAPI `BackgroundTasks` (in-process)

---

## 3. Data Model

### 3.1 Entity-Relationship Diagram

```
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│  Document    │──1:N─│    Chunk     │──1:N─│   Evidence   │
│              │      │              │      │              │
│ pmid         │      │ doc_id (FK)  │      │ chunk_id(FK) │
│ title        │      │ text         │      │ relation_id  │
│ abstract     │      │ char_start   │      │ quote        │
│ authors      │      │ char_end     │      │ quote_start  │
│ pub_date     │      │ token_count  │      │ quote_end    │
│ mesh_terms   │      │ processed    │      │ quote_hash   │
└──────────────┘      └──────────────┘      └──────┬───────┘
                                                   │
                      ┌──────────────┐      ┌──────▼───────┐
                      │   Entity     │──N:M─│   Relation   │
                      │              │      │              │
                      │ name         │      │ head_id (FK) │
                      │ type (enum)  │      │ tail_id (FK) │
                      │ norm_name    │      │ label (enum) │
                      │ aliases[]    │      │ confidence   │
                      │ description  │      │ properties{} │
                      └──────────────┘      └──────────────┘
```

### 3.2 Controlled Vocabulary

All entity types and relation labels are enforced by PostgreSQL enums, preventing schema drift from LLM hallucinations.

**Entity Types (9):**

| Type | Description | Examples |
|------|-------------|----------|
| Disease | Medical conditions | Parkinson's disease, Gaucher disease |
| Gene | Gene names/symbols | GBA1, SNCA, LRRK2 |
| Protein | Proteins/enzymes | GCase, α-synuclein |
| Compound/Drug | Drugs/chemicals | Ambroxol, levodopa |
| Pathway | Biological pathways | Autophagy, UPR |
| CellType | Cell types | Dopaminergic neurons, microglia |
| Tissue/Region | Anatomical regions | Substantia nigra, brain |
| Phenotype | Observable traits | Motor symptoms, tremor |
| Biomarker | Biological markers | CSF α-synuclein levels |

**Relation Labels (11):**

| Label | Direction | Example |
|-------|-----------|---------|
| ASSOCIATED_WITH | Bidirectional | GBA1 ↔ Parkinson's disease |
| INCREASES_RISK | Directional | GBA1 mutation → Parkinson's |
| DECREASES_RISK | Directional | Ambroxol → Parkinson's |
| UPREGULATES | Directional | Ambroxol → GCase activity |
| DOWNREGULATES | Directional | α-synuclein → lysosomal function |
| ACTIVATES | Directional | Ambroxol → GCase |
| INHIBITS | Directional | α-synuclein aggregation → autophagy |
| BINDS | Directional | Ambroxol → GCase |
| BIOMARKER_FOR | Directional | GCase activity → Gaucher disease |
| PART_OF | Directional | Substantia nigra → brain |
| EXPRESSED_IN | Directional | GBA1 → dopaminergic neurons |

### 3.3 Key Design Decisions

**UUID7 Primary Keys** — All tables use UUID7 (time-sortable UUIDs) for globally unique, time-ordered identifiers without database coordination. This enables future sharding while maintaining insert ordering.

**Entity Deduplication via `normalized_name`** — Entities are deduplicated by their lowercase, underscore-joined normalized name. "Ambroxol", "ambroxol hydrochloride", and "AMB" all map to the same entity. Subsequent mentions are merged into the aliases array.

**Evidence as First-Class Citizen** — Every relation must have at least one evidence record containing the source quote and character offsets. Evidence records have a `quote_hash` (SHA256) to prevent duplicate storage while enabling efficient lookups.

**Idempotent Pipeline** — The extraction pipeline uses `processed` flags on chunks and hash-based uniqueness constraints on evidence. Re-running extraction on the same data is safe and produces no duplicates.

---

## 4. Extraction Pipeline

### 4.1 Pipeline Stages

```
Stage 1: PubMed Fetch          Stage 2: Chunking           Stage 3: LLM Extraction
─────────────────────         ─────────────────         ─────────────────────────
  NCBI E-utilities              Split text                Gemini 2.5 Flash
  ↓                             ↓                         ↓
  Search "ambroxol              Sentence-aware            Entity extraction
  parkinson"                    1000-char chunks           (JSON prompt)
  ↓                             200-char overlap          ↓
  Fetch XML metadata            ↓                         Relation extraction
  ↓                             Track char offsets          (JSON prompt)
  Parse & store                 ↓                         ↓
  documents                     Store chunks              Upsert entities
                                                          Upsert relations
                                                          Store evidence
```

### 4.2 LLM Integration

The extraction uses a two-pass approach per chunk:

1. **Entity Pass** — Extract all biomedical entities with types and normalized names
2. **Relation Pass** — Given the entities, extract relationships with confidence scores and evidence quotes

Both passes use structured JSON prompts with the controlled vocabulary embedded in the system message. The LLM is instructed to respond with only valid JSON, which is parsed and validated against the schema enums.

**LLM Client Abstraction:** The `BaseLLMClient` abstract class allows swapping between providers:

- `GeminiClient` — Production client using Google Gemini 2.5 Flash via REST API
- `MockLLMClient` — Deterministic test client returning predefined responses

### 4.3 Chunking Strategy

Documents are split into chunks using sentence-aware boundaries:

- **Chunk size**: 1000 characters (configurable)
- **Overlap**: 200 characters (configurable)
- **Splitting**: Regex-based sentence boundary detection, falling back to paragraph breaks
- **Offset tracking**: Each chunk records `char_start` and `char_end` relative to the full document text, enabling evidence quote positioning

### 4.4 Error Handling

- **LLM parse failures**: Logged and skipped (chunk marked unprocessed for retry)
- **Schema violations**: Entities/relations with invalid types are silently dropped
- **Rate limits**: Exponential backoff with jitter (tenacity library)
- **Partial failures**: Each chunk is independent; failures don't block other chunks

---

## 5. API Design

### 5.1 Endpoint Summary

All endpoints are prefixed with `/api/v1/`.

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/documents/fetch-pubmed` | Trigger PubMed fetch |
| `GET` | `/documents/` | List documents (paginated) |
| `GET` | `/documents/stats` | Document statistics |
| `GET` | `/documents/{id}` | Get single document |
| `DELETE` | `/documents/{id}` | Delete document + cascade |
| `GET` | `/entities/` | Search entities |
| `GET` | `/entities/types` | List entity type counts |
| `GET` | `/entities/{id}` | Get entity with relations |
| `GET` | `/entities/{id}/relations` | Entity's relations |
| `GET` | `/entities/{id}/neighborhood` | N-hop neighborhood |
| `GET` | `/graph/subgraph` | Subgraph around entity |
| `GET` | `/graph/paths` | Shortest paths between entities |
| `GET` | `/graph/stats` | Graph-wide statistics |
| `POST` | `/jobs/extract` | Start extraction job |
| `GET` | `/jobs/` | List jobs |
| `GET` | `/jobs/{id}` | Job status + progress |
| `POST` | `/jobs/{id}/cancel` | Cancel running job |
| `GET` | `/jobs/stats` | Job statistics |

### 5.2 Query Patterns

**Entity Search** supports filtering by name (substring), type, and pagination:
```
GET /api/v1/entities/?q=ambroxol&type=Compound/Drug&limit=10
```

**Neighborhood Traversal** returns N-hop ego graph from a starting entity:
```
GET /api/v1/entities/{id}/neighborhood?depth=2&max_nodes=50
```

**Path Finding** discovers shortest paths between two entities using BFS:
```
GET /api/v1/graph/paths?source_id=...&target_id=...&max_depth=4
```

### 5.3 Response Design

All responses follow a consistent structure with Pydantic schemas. Relations include evidence records inline, enabling clients to display provenance:

```json
{
  "id": "019...",
  "label": "ACTIVATES",
  "head_entity": {"name": "Ambroxol", "type": "Compound/Drug"},
  "tail_entity": {"name": "GCase", "type": "Protein"},
  "confidence": 0.92,
  "evidence": [
    {
      "quote": "Ambroxol enhances GCase enzymatic activity...",
      "chunk_id": "019..."
    }
  ]
}
```

---

## 6. Database Design

### 6.1 Technology Choice

**PostgreSQL 16** was chosen over graph-specific databases (Neo4j) because:

- The dataset is moderate-scale (hundreds of entities, thousands of relations)
- SQL joins efficiently handle 2–3 hop traversals needed for this domain
- Alembic migrations provide schema evolution
- Full ACID compliance for concurrent pipeline writes
- No additional infrastructure or licensing costs

### 6.2 Indexing Strategy

| Index | Table | Purpose |
|-------|-------|---------|
| `ix_entities_normalized_name` | entities | Deduplication lookups |
| `ix_entities_type` | entities | Type-filtered queries |
| `ix_relations_head_id` | relations | Outgoing traversal |
| `ix_relations_tail_id` | relations | Incoming traversal |
| `ix_relations_label` | relations | Label-filtered queries |
| `ix_evidence_relation_id` | evidence | Evidence per relation |
| `ix_evidence_quote_hash` | evidence | Deduplication |
| `ix_chunks_document_id` | chunks | Document's chunks |
| `ix_chunks_processed` | chunks | Pipeline progress |

### 6.3 Cascade Behavior

`ON DELETE CASCADE` is set on all foreign keys:

- Deleting a document removes its chunks and downstream evidence
- Deleting an entity removes its relations and their evidence
- This enables clean data resets without orphan records

---

## 7. Testing Strategy

### 7.1 Test Structure

```
tests/
├── conftest.py              # Shared fixtures (DB session, test client)
├── unit/
│   ├── test_config.py       # Configuration validation (12 tests)
│   └── test_extraction.py   # Extraction logic (25 tests)
└── integration/
    └── test_api.py          # Full API endpoint tests (26 tests)
```

**Total: 63 tests, all passing.**

### 7.2 Integration Tests

Integration tests run against a live PostgreSQL instance using async SQLAlchemy sessions. They cover:

- Document CRUD and PubMed fetch
- Entity search, filtering, and pagination
- Graph traversal (neighborhood, subgraph, paths)
- Job lifecycle (create, list, cancel)
- Error cases (404s, invalid inputs)

### 7.3 Unit Tests

Unit tests verify extraction logic in isolation:

- Entity type enum validation and fuzzy matching
- Relation label validation
- JSON parse error handling
- Confidence threshold filtering

---

## 8. Evaluation & Visualization

### 8.1 Evaluation Harness (`scripts/eval.py`)

The evaluation harness runs 7 domain-specific sanity checks:

1. Ambroxol entity exists with relations
2. Parkinson's disease entity exists with relations
3. Key mechanistic entities present (GCase, α-synuclein, lysosome)
4. Entity type diversity (multiple types represented)
5. Relation label diversity (multiple labels used)
6. Evidence validity (quotes are substrings of source chunks)
7. Ambroxol-Parkinson's direct connection exists

It also computes quality metrics: duplicate detection, orphan relations, evidence coverage, and confidence statistics.

### 8.2 Interactive Visualization (`scripts/visualize.py`)

Generates an interactive HTML graph using pyvis (vis.js) with:

- **Dual-center default** — Expands from both Ambroxol and Parkinson's simultaneously
- **Depth control** — +/− buttons to expand or collapse from center nodes
- **Entity type toggles** — Click buttons to show/hide entity types
- **Relation type toggles** — Click buttons to filter relation types (auto-hides orphaned nodes)
- **Evidence on hover** — Hovering edges shows confidence and evidence quotes
- **Static layout** — Physics runs once to compute positions, then freezes for stable viewing

---

## 9. Configuration

All configuration is via environment variables (12-factor app), loaded through Pydantic Settings with `.env` file support:

| Variable | Default | Purpose |
|----------|---------|---------|
| `POSTGRES_HOST` | localhost | Database host |
| `POSTGRES_PORT` | 5432 | Database port |
| `POSTGRES_USER` | lifeai | Database user |
| `POSTGRES_PASSWORD` | lifeai_dev_password | Database password |
| `POSTGRES_DB` | lifeai_kg | Database name |
| `GOOGLE_API_KEY` | — | Gemini API key (required for extraction) |
| `GEMINI_MODEL` | gemini-2.5-flash | Model for extraction |
| `LLM_EXTRACTOR` | gemini | Extractor backend (gemini/mock) |
| `CHUNK_SIZE` | 1000 | Characters per chunk |
| `CHUNK_OVERLAP` | 200 | Overlap between chunks |
| `LOG_LEVEL` | INFO | Logging verbosity |

---

## 10. Future Work

- **Full extraction completion** — Run pipeline on all 912 chunks (currently ~5%)
- **Entity resolution** — ML-based coreference to merge entity variants more aggressively
- **Vector embeddings** — Add semantic search via pgvector
- **Graph algorithms** — PageRank, community detection for entity importance
- **Incremental updates** — Schedule daily PubMed ingestion for new publications
- **Frontend** — React dashboard with interactive graph exploration
