"""
Workers module — Celery tasks for async KG extraction.

Architecture:
    FastAPI API  ──dispatch──>  Redis Queue  ──consume──>  Celery Worker
                                                              │
    Redis Job Store  <──progress updates──────────────────────┘

Start worker:
    celery -A src.workers.celery_app worker -l info -P solo -Q kg_extraction

The -P solo pool is required because tasks use asyncio.run() internally.
"""

from src.workers.celery_app import celery_app
from src.workers.job_store import create_job, get_job, list_jobs, update_job

__all__ = [
    "celery_app",
    "create_job",
    "get_job",
    "list_jobs",
    "update_job",
]
