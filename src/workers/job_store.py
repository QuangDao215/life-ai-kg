"""
Redis-backed job store for tracking extraction jobs.

This store is shared between the FastAPI process and Celery workers,
enabling the API to report live progress while the worker runs.

Falls back to in-memory storage if Redis is unavailable (for testing).

Requirements addressed:
- REQ-KG-06: Job status persistence with progress and error details
"""

import json
from datetime import datetime
from uuid import UUID

from src.core.config import settings
from src.core.logging import get_logger
from src.schemas.jobs import JobStatus, JobType

logger = get_logger(__name__)

# Redis key prefix
JOB_PREFIX = "lifeai:job:"
JOB_INDEX_KEY = "lifeai:jobs"

# In-memory fallback
_memory_store: dict[str, dict] = {}
_memory_index: list[str] = []

# Redis client (lazy init)
_redis_client = None
_redis_available: bool | None = None


def _get_redis():
    """Get or create Redis client. Returns None if unavailable."""
    global _redis_client, _redis_available

    if _redis_available is False:
        return None

    if _redis_client is not None:
        return _redis_client

    try:
        import redis
        _redis_client = redis.Redis.from_url(
            settings.redis_dsn,
            decode_responses=True,
            socket_connect_timeout=2,
        )
        _redis_client.ping()
        _redis_available = True
        logger.info("Job store: using Redis", url=settings.redis_dsn)
        return _redis_client
    except Exception as e:
        _redis_available = False
        logger.warning("Job store: Redis unavailable, using in-memory fallback", error=str(e))
        return None


def _serialize_job(job: dict) -> str:
    """Serialize job dict to JSON string."""
    serializable = {}
    for k, v in job.items():
        if isinstance(v, UUID):
            serializable[k] = str(v)
        elif isinstance(v, datetime):
            serializable[k] = v.isoformat()
        elif isinstance(v, (JobStatus, JobType)):
            serializable[k] = v.value
        else:
            serializable[k] = v
    return json.dumps(serializable)


def _deserialize_job(data: str) -> dict:
    """Deserialize JSON string to job dict."""
    job = json.loads(data)

    # Restore types
    if job.get("id"):
        job["id"] = UUID(job["id"])
    if job.get("status"):
        job["status"] = JobStatus(job["status"])
    if job.get("job_type"):
        job["job_type"] = JobType(job["job_type"])
    for ts_field in ("created_at", "started_at", "completed_at"):
        if job.get(ts_field):
            job[ts_field] = datetime.fromisoformat(job[ts_field])

    return job


def create_job(job_type: JobType) -> dict:
    """Create a new job record."""
    from uuid6 import uuid7

    job_id = uuid7()
    job = {
        "id": job_id,
        "job_type": job_type,
        "status": JobStatus.PENDING,
        "progress": 0.0,
        "total_items": 0,
        "processed_items": 0,
        "error_count": 0,
        "error_message": None,
        "result": None,
        "created_at": datetime.utcnow(),
        "started_at": None,
        "completed_at": None,
    }

    r = _get_redis()
    key = f"{JOB_PREFIX}{job_id}"

    if r:
        r.set(key, _serialize_job(job), ex=86400)  # 24h TTL
        r.lpush(JOB_INDEX_KEY, str(job_id))
        r.ltrim(JOB_INDEX_KEY, 0, 999)  # Keep last 1000 jobs
    else:
        _memory_store[str(job_id)] = job
        _memory_index.insert(0, str(job_id))

    return job


def update_job(job_id: UUID, **updates) -> dict | None:
    """Update a job record."""
    job = get_job(job_id)
    if not job:
        return None

    job.update(updates)

    r = _get_redis()
    key = f"{JOB_PREFIX}{job_id}"

    if r:
        r.set(key, _serialize_job(job), ex=86400)
    else:
        _memory_store[str(job_id)] = job

    return job


def get_job(job_id: UUID) -> dict | None:
    """Get a job record by ID."""
    r = _get_redis()
    key = f"{JOB_PREFIX}{job_id}"

    if r:
        data = r.get(key)
        if data:
            return _deserialize_job(data)
        return None
    else:
        return _memory_store.get(str(job_id))


def list_jobs(
    status_filter: JobStatus | None = None,
    offset: int = 0,
    limit: int = 20,
) -> tuple[list[dict], int]:
    """List jobs with optional status filter. Returns (jobs, total)."""
    r = _get_redis()

    if r:
        # Get all job IDs from index
        all_ids = r.lrange(JOB_INDEX_KEY, 0, -1)
        jobs = []
        for jid in all_ids:
            data = r.get(f"{JOB_PREFIX}{jid}")
            if data:
                jobs.append(_deserialize_job(data))
    else:
        jobs = [
            _memory_store[jid]
            for jid in _memory_index
            if jid in _memory_store
        ]

    # Filter
    if status_filter:
        jobs = [j for j in jobs if j["status"] == status_filter]

    # Sort by created_at desc (newest first)
    jobs.sort(key=lambda j: j.get("created_at", datetime.min), reverse=True)

    total = len(jobs)
    page = jobs[offset: offset + limit]

    return page, total
