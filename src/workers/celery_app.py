"""
Celery application configuration.

This module creates and configures the Celery app instance used by
the KG extraction worker.

Usage:
    # Start worker:
    celery -A src.workers.celery_app worker -l info -P solo

    # -P solo is required because we use asyncio inside tasks
"""

from celery import Celery

from src.core.config import settings

# Create Celery app
celery_app = Celery(
    "life_ai_kg",
    broker=settings.celery_broker,
    backend=settings.celery_backend,
)

# Configuration
celery_app.conf.update(
    # Serialization
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",

    # Timezone
    timezone="UTC",
    enable_utc=True,

    # Task settings
    task_track_started=True,
    task_acks_late=True,  # Acknowledge after task completes (reliability)
    worker_prefetch_multiplier=1,  # One task at a time (LLM calls are slow)

    # Eager mode: execute tasks synchronously in-process (for testing)
    task_always_eager=settings.celery_task_always_eager,

    # Result settings
    result_expires=86400,  # 24 hours

    # Task routes
    task_routes={
        "src.workers.tasks.*": {"queue": "kg_extraction"},
    },

    # Default queue
    task_default_queue="kg_extraction",
)

# Auto-discover tasks
celery_app.autodiscover_tasks(["src.workers"])
