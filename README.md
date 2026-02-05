# LIFE AI Knowledge Graph Backend

A biomedical Knowledge Graph pipeline that collects PubMed literature on Ambroxol–Parkinson's disease, extracts entities and relationships using Google Gemini, and serves them through a REST API with interactive visualization.

```
PubMed → Chunking → Gemini LLM → PostgreSQL KG → FastAPI → Interactive Graph
```

---

## Quick Start

### Prerequisites

- **Python 3.11+** — [python.org/downloads](https://www.python.org/downloads/)
- **Docker Desktop** — [docker.com](https://www.docker.com/products/docker-desktop/) (for PostgreSQL + Redis)
- **Google API Key** — [aistudio.google.com](https://aistudio.google.com/apikey) (for Gemini extraction)

### One-Command Setup

```powershell
# Windows
.\setup.ps1

# Linux/Mac
chmod +x setup.sh && ./setup.sh
```

This creates a virtual environment, installs dependencies, starts Docker containers, runs migrations, and verifies the setup.

### Manual Setup

```bash
# 1. Virtual environment
python -m venv venv
source venv/bin/activate          # Windows: .\venv\Scripts\Activate.ps1
pip install -e ".[dev]"

# 2. Infrastructure
docker-compose up -d postgres redis

# 3. Environment
cp .env.example .env
# Edit .env → set GOOGLE_API_KEY=your_key_here

# 4. Database
alembic upgrade head

# 5. API server
uvicorn src.main:app --reload --port 8000

# 6. Celery worker (separate terminal — required for queue-based extraction)
celery -A src.workers.celery_app worker -l info -P solo -Q kg_extraction
```

> **Note:** The Celery worker picks extraction jobs from the Redis queue. If Redis/Celery
> is unavailable, the API automatically falls back to in-process execution.

Verify: `curl http://localhost:8000/health` or open http://localhost:8000/docs

### One-Command Demo

After setup, run the full pipeline end-to-end:

```powershell
.\run-demo.ps1
```

This automatically: starts Docker + API + Celery worker → fetches PubMed articles → chunks documents → dispatches KG extraction via Celery queue → polls job progress → runs 6 graph exploration queries → generates evaluation report → produces interactive visualization. All outputs are saved to `demo_output/` and `examples/`.

---

## End-to-End Pipeline

### Step 1 — Fetch PubMed Articles

```bash
python scripts/fetch_pubmed.py --query "ambroxol parkinson" --max-results 200
```

Downloads abstracts and metadata from PubMed E-utilities. Stores documents with title, abstract, authors, MeSH terms, and publication date.

### Step 2 — Chunk Documents

```bash
python scripts/chunk_documents.py
```

Splits document texts into ~1000-character chunks with 200-character overlap. Tracks character offsets for evidence provenance.

### Step 3 — Extract Knowledge Graph

**Via API (primary — uses Celery queue/worker):**

```bash
# Start worker in a separate terminal
celery -A src.workers.celery_app worker -l info -P solo -Q kg_extraction

# Trigger extraction
curl -X POST http://localhost:8000/api/v1/jobs/kg-build \
  -H "Content-Type: application/json" \
  -d '{"provider": "gemini", "limit": 20, "skip_processed": true}'

# Poll progress
curl http://localhost:8000/api/v1/jobs/{JOB_ID}
```

**Via CLI (convenience):**

```bash
python scripts/extract_kg.py --provider gemini
```

Runs a two-pass LLM extraction per chunk:
1. **Entity pass** — Extracts biomedical entities (drugs, genes, proteins, diseases, etc.)
2. **Relation pass** — Extracts typed relationships between entities with confidence scores and evidence quotes

Progress is saved per-chunk; safe to interrupt and resume.

### Step 4 — Explore

```bash
# Start API
uvicorn src.main:app --reload --port 8000

# Generate visualization
python scripts/visualize.py

# Run evaluation
python scripts/eval.py
```

---

## API Reference

Base URL: `http://localhost:8000/api/v1`

Interactive docs: http://localhost:8000/docs

### Documents

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/documents` | Create a document |
| `GET` | `/documents` | List documents (paginated) |
| `GET` | `/documents/{id}` | Get document by ID |
| `GET` | `/documents/{id}/chunks` | Get document's text chunks |
| `DELETE` | `/documents/{id}` | Delete document (cascades) |

### Entities

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/entities/search?query=ambroxol` | Search entities by name |
| `GET` | `/entities/types` | Entity type counts |
| `GET` | `/entities/{id}` | Get entity with details |
| `GET` | `/entities/{id}/neighborhood?limit=50` | Direct neighborhood graph |
| `GET` | `/entities/{id}/relations` | Entity's relations with evidence |

### Graph

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/graph/subgraph?entity_id=...&depth=2` | Subgraph around entity |
| `GET` | `/graph/path?source_id=...&target_id=...` | Shortest path between entities |
| `GET` | `/graph/stats` | Graph-wide statistics |

### Jobs

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/jobs/kg-build` | Trigger KG extraction job |
| `GET` | `/jobs` | List all jobs |
| `GET` | `/jobs/{id}` | Job status and progress |
| `POST` | `/jobs/{id}/cancel` | Cancel a running job |
| `GET` | `/jobs/progress/current` | Live extraction progress |

### Example Queries

```bash
# Search for ambroxol entities
curl "http://localhost:8000/api/v1/entities/search?query=ambroxol&page_size=5"

# Get entity neighborhood
curl "http://localhost:8000/api/v1/entities/{ENTITY_ID}/neighborhood?limit=50"

# Find path between two entities
curl "http://localhost:8000/api/v1/graph/path?source_id={ID1}&target_id={ID2}&max_hops=4"

# Graph statistics
curl "http://localhost:8000/api/v1/graph/stats"
```

---

## Visualization

```bash
python scripts/visualize.py
# Opens: kg_visualization.html
```

Generates an interactive HTML graph (vis.js) with:

- **Dual-center view** — Expands from both Ambroxol and Parkinson's by default
- **Depth control** — `−` / `+` buttons to expand or collapse from center nodes
- **Entity type buttons** — Click to show/hide Disease, Drug, Gene, Protein, etc.
- **Relation type buttons** — Click to toggle relation types; orphaned nodes auto-hide
- **Evidence tooltips** — Hover edges to see confidence and source quotes
- **Static layout** — Physics computes initial positions then freezes for stable viewing
- **Draggable nodes** — Reposition individual nodes by dragging

```bash
# Custom centers
python scripts/visualize.py --centers "gba1,ambroxol,alpha-synuclein"

# Deeper traversal
python scripts/visualize.py --depth 3 --max-nodes 200

# Full graph
python scripts/visualize.py --full
```

---

## Evaluation

```bash
python scripts/eval.py
```

Runs 7 domain-specific sanity checks:

1. Ambroxol entity exists with relations
2. Parkinson's disease entity exists with relations
3. Key mechanistic entities present (GCase, α-synuclein, lysosome)
4. Entity type diversity (≥4 types represented)
5. Relation label diversity (≥3 labels used)
6. Evidence validity (quotes found in source chunks)
7. Direct Ambroxol–Parkinson's connection exists

Also computes quality metrics: duplicate detection, orphan analysis, evidence coverage, and confidence statistics.

**Pre-generated artifacts** (from `.\run-demo.ps1`):

- `examples/EVAL_REPORT.md` — Full evaluation report with all sanity checks and metrics
- `examples/api_responses/` — Saved API responses from graph exploration endpoints (entity search, neighborhood, subgraph, path finding, job status)
- `examples/kg_visualization.html` — Interactive graph visualization (open in browser)

---

## Testing

```bash
# All tests (63 passing)
pytest tests/ -v

# Unit tests only
pytest tests/unit/ -v

# Integration tests (requires running PostgreSQL)
pytest tests/integration/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

Test breakdown:
- `tests/unit/test_config.py` — Configuration validation (12 tests)
- `tests/unit/test_extraction.py` — Extraction logic, enum matching (25 tests)
- `tests/integration/test_api.py` — Full API endpoint coverage (26 tests)

---

## Project Structure

```
life-ai-kg/
├── src/
│   ├── api/                    # FastAPI route handlers
│   │   ├── documents.py        #   Document CRUD + PubMed fetch
│   │   ├── entities.py         #   Entity search + neighborhood
│   │   ├── graph.py            #   Subgraph, path finding, stats
│   │   └── jobs.py             #   Extraction job management
│   ├── core/
│   │   ├── config.py           # Pydantic Settings (env vars)
│   │   └── logging.py          # Structured logging (structlog)
│   ├── db/
│   │   ├── models/
│   │   │   ├── document.py     # PubMed publication
│   │   │   ├── chunk.py        # Text chunk with offsets
│   │   │   ├── entity.py       # KG node (drug, gene, etc.)
│   │   │   ├── relation.py     # KG edge (activates, inhibits, etc.)
│   │   │   ├── evidence.py     # Source quote for a relation
│   │   │   ├── job.py          # Extraction job tracking
│   │   │   └── audit_log.py    # Change audit trail
│   │   ├── enums.py            # EntityType, RelationLabel enums
│   │   ├── base.py             # SQLAlchemy base, UUID7 mixin
│   │   └── session.py          # Async session factory
│   ├── workers/
│   │   ├── celery_app.py       # Celery app configuration
│   │   ├── tasks.py            # KG extraction Celery task
│   │   └── job_store.py        # Redis-backed job state store
│   ├── schemas/                # Pydantic request/response models
│   ├── services/
│   │   ├── pubmed.py           # PubMed E-utilities async client
│   │   ├── chunking.py         # Sentence-aware text splitter
│   │   ├── extraction.py       # LLM entity/relation extraction
│   │   └── llm_client.py       # Gemini client + mock for tests
│   └── main.py                 # FastAPI app factory
├── scripts/
│   ├── fetch_pubmed.py         # CLI: fetch PubMed articles
│   ├── chunk_documents.py      # CLI: chunk all documents
│   ├── extract_kg.py           # CLI: run KG extraction pipeline
│   ├── visualize.py            # CLI: generate interactive graph HTML
│   └── eval.py                 # CLI: run evaluation harness
├── tests/
│   ├── unit/                   # Fast tests, no DB needed
│   └── integration/            # API tests against live DB
├── examples/
│   ├── api_responses/          # Saved API response examples (auto-generated)
│   ├── EVAL_REPORT.md          # Evaluation report (auto-generated)
│   └── kg_visualization.html   # Interactive graph (auto-generated)
├── alembic/                    # Database migrations
├── docker-compose.yml          # PostgreSQL 16 + Redis 7
├── pyproject.toml              # Dependencies + tool config
├── Makefile                    # Development shortcuts
├── setup.ps1 / setup.sh       # One-command setup
├── run-demo.ps1 / run-demo.sh # One-command demo
├── reset.ps1                  # Stop services + clean data
└── DESIGN.md                   # Technical design document
```

---

## Configuration

All config via environment variables (`.env` file supported):

| Variable | Default | Description |
|----------|---------|-------------|
| `POSTGRES_HOST` | `localhost` | PostgreSQL host |
| `POSTGRES_PORT` | `5432` | PostgreSQL port |
| `POSTGRES_USER` | `lifeai` | Database user |
| `POSTGRES_PASSWORD` | `lifeai_dev_password` | Database password |
| `POSTGRES_DB` | `lifeai_kg` | Database name |
| `GOOGLE_API_KEY` | — | Google Gemini API key (**required** for extraction) |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Gemini model name |
| `LLM_EXTRACTOR` | `gemini` | Backend: `gemini` or `mock` |
| `CHUNK_SIZE` | `1000` | Characters per text chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between adjacent chunks |
| `REDIS_HOST` | `localhost` | Redis host (for Celery broker + job store) |
| `REDIS_PORT` | `6379` | Redis port |
| `CELERY_BROKER_URL` | `redis://localhost:6379/0` | Celery message broker |
| `CELERY_RESULT_BACKEND` | `redis://localhost:6379/1` | Celery result backend |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

---

## Make Targets

```bash
make help           # Show all available commands
make docker-up      # Start PostgreSQL + Redis
make docker-down    # Stop containers
make db-upgrade     # Apply migrations
make test           # Run all tests
make test-unit      # Unit tests only
make lint           # Ruff linter
make format         # Ruff formatter
make run-api        # Start FastAPI server
make run-worker     # Start Celery worker
make eval           # Run evaluation harness
make clean          # Remove caches
```

---

## Database Reset

Use `reset.ps1` to clean project state at three levels:

```powershell
.\reset.ps1              # Basic: stop processes + remove generated files (keep DB)
.\reset.ps1 -WipeDB      # + wipe PostgreSQL and Redis volumes
.\reset.ps1 -Full         # + remove venv/ and .env (fresh clone state)
```

After reset:

| Mode | Next steps |
|------|------------|
| Basic | `.\run-demo.ps1` |
| `-WipeDB` | `docker-compose up -d` → `alembic upgrade head` → `.\run-demo.ps1` |
| `-Full` | `.\setup.ps1` → edit `.env` → `.\run-demo.ps1` |

---

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| API | FastAPI + Uvicorn | Async REST API with auto-generated OpenAPI docs |
| Database | PostgreSQL 16 | Relational storage with enum constraints |
| ORM | SQLAlchemy 2.0 (async) | Async database access with type safety |
| Migrations | Alembic | Schema versioning |
| LLM | Google Gemini 2.5 Flash | Entity and relation extraction |
| Queue | Redis + Celery | Async job processing |
| Validation | Pydantic v2 | Request/response schemas |
| Visualization | pyvis (vis.js) | Interactive graph rendering |
| Testing | pytest + pytest-asyncio | Async test support |
| HTTP | httpx + tenacity | Async HTTP with retry logic |

---

## Architecture

See [DESIGN.md](DESIGN.md) for detailed architecture decisions, data model, extraction pipeline design, and evaluation strategy.

## License

MIT
