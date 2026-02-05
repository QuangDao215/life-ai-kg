# LIFE AI Knowledge Graph - Development Commands
# Usage: make <target>

.PHONY: help install dev-install docker-up docker-down db-migrate db-upgrade db-downgrade \
        test test-unit test-integration lint format typecheck pre-commit \
        run-api run-worker fetch-pubmed eval clean

# Default target
help:
	@echo "LIFE AI Knowledge Graph - Available Commands"
	@echo ""
	@echo "Setup:"
	@echo "  install        Install production dependencies"
	@echo "  dev-install    Install all dependencies including dev tools"
	@echo "  docker-up      Start PostgreSQL and Redis containers"
	@echo "  docker-down    Stop and remove containers"
	@echo ""
	@echo "Database:"
	@echo "  db-migrate     Create a new migration (MSG=description)"
	@echo "  db-upgrade     Apply all pending migrations"
	@echo "  db-downgrade   Rollback last migration"
	@echo ""
	@echo "Testing:"
	@echo "  test           Run all tests"
	@echo "  test-unit      Run unit tests only"
	@echo "  test-int       Run integration tests only"
	@echo "  test-cov       Run tests with coverage report"
	@echo ""
	@echo "Code Quality:"
	@echo "  lint           Run linter (ruff)"
	@echo "  format         Format code (ruff format)"
	@echo "  typecheck      Run type checker (mypy)"
	@echo "  pre-commit     Run all pre-commit hooks"
	@echo ""
	@echo "Running:"
	@echo "  run-api        Start FastAPI server"
	@echo "  run-worker     Start Celery worker"
	@echo ""
	@echo "Data Pipeline:"
	@echo "  fetch-pubmed   Fetch PubMed data"
	@echo "  eval           Run evaluation harness"
	@echo ""
	@echo "Maintenance:"
	@echo "  clean          Remove cache and build artifacts"

# ============================================
# Setup
# ============================================
install:
	pip install -e .

dev-install:
	pip install -e ".[dev]"
	pre-commit install

docker-up:
	docker-compose up -d postgres redis
	@echo "Waiting for services to be healthy..."
	@sleep 3
	@docker-compose ps

docker-down:
	docker-compose down

docker-reset:
	docker-compose down -v
	docker-compose up -d postgres redis

# ============================================
# Database
# ============================================
db-migrate:
	@if [ -z "$(MSG)" ]; then \
		echo "Error: Please provide a migration message with MSG=<description>"; \
		exit 1; \
	fi
	alembic revision --autogenerate -m "$(MSG)"

db-upgrade:
	alembic upgrade head

db-downgrade:
	alembic downgrade -1

db-reset:
	alembic downgrade base
	alembic upgrade head

# ============================================
# Testing
# ============================================
test:
	pytest tests/ -v

test-unit:
	pytest tests/unit/ -v -m unit

test-int:
	pytest tests/integration/ -v -m integration

test-cov:
	pytest tests/ --cov=src --cov-report=html --cov-report=term-missing

# ============================================
# Code Quality
# ============================================
lint:
	ruff check src/ tests/

lint-fix:
	ruff check src/ tests/ --fix

format:
	ruff format src/ tests/

format-check:
	ruff format src/ tests/ --check

typecheck:
	mypy src/

pre-commit:
	pre-commit run --all-files

# ============================================
# Running Services
# ============================================
run-api:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

run-worker:
	celery -A src.workers.celery_app worker --loglevel=info -P solo -Q kg_extraction

run-worker-beat:
	celery -A src.workers.celery_app beat --loglevel=info

# ============================================
# Data Pipeline
# ============================================
fetch-pubmed:
	python -m scripts.fetch_pubmed

eval:
	python -m scripts.eval

# ============================================
# Maintenance
# ============================================
clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
