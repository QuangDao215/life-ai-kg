"""
Job API endpoints for async KG build operations.

Architecture:
    POST /jobs/kg-build
        → Creates job in Redis-backed store
        → Dispatches Celery task to Redis queue
        → Returns 202 Accepted immediately

    Celery Worker (separate process)
        → Picks task from Redis queue
        → Runs LLM extraction
        → Updates job progress in Redis store

    GET /jobs/{id}
        → Reads current job state from Redis store
        → Returns live progress even while worker is running

Fallback:
    If Celery/Redis is unavailable (e.g., local dev without Docker),
    falls back to FastAPI BackgroundTasks (in-process, same behavior
    but no separate worker process).

Requirements addressed:
- Assignment: "KG build pipeline must run asynchronously using a queue/worker architecture"
- REQ-KG-06: Job status tracking with progress and error details
"""

from datetime import datetime
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.config import settings
from src.core.logging import get_logger
from src.db import get_db
from src.db.models import Chunk, Entity, Evidence, Relation
from src.schemas import (
    ExtractionProgress,
    JobCreateResponse,
    JobResponse,
    JobStatus,
    JobSummary,
    JobType,
    KGBuildRequest,
    PaginatedResponse,
)
from src.workers.job_store import create_job, get_job, list_jobs, update_job

logger = get_logger(__name__)

router = APIRouter()


# =============================================================================
# Celery Dispatch Helpers
# =============================================================================


def _celery_available() -> bool:
    """Check if Celery broker (Redis) is reachable."""
    if settings.celery_task_always_eager:
        return False
    try:
        from src.workers.celery_app import celery_app
        conn = celery_app.connection()
        conn.ensure_connection(max_retries=1, timeout=2)
        conn.close()
        return True
    except Exception:
        return False


def _dispatch_to_celery(
    job_id: UUID,
    request: KGBuildRequest,
) -> str | None:
    """
    Dispatch KG build task to Celery queue.

    Returns the Celery task ID on success, None on failure.
    """
    try:
        from src.workers.tasks import kg_build_task

        result = kg_build_task.delay(
            job_id_str=str(job_id),
            provider=request.provider,
            limit=request.limit,
            skip_processed=request.skip_processed,
        )
        return result.id
    except Exception as e:
        logger.warning("Failed to dispatch to Celery", error=str(e))
        return None


# =============================================================================
# BackgroundTasks Fallback (when Celery is unavailable)
# =============================================================================


async def _fallback_kg_build_task(
    job_id: UUID,
    request: KGBuildRequest,
    db_url: str,
) -> None:
    """
    In-process fallback for KG extraction when Celery is unavailable.

    Same logic as the Celery task but runs inside the FastAPI process
    via BackgroundTasks. Used for local dev without Docker/Redis.
    """
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    from src.db.models import Chunk, Evidence
    from src.services.extraction import ExtractionService
    from src.services.llm_client import get_llm_client

    engine = create_async_engine(db_url, echo=False)
    async_session = async_sessionmaker(engine, expire_on_commit=False)

    try:
        update_job(job_id, status=JobStatus.RUNNING, started_at=datetime.utcnow())

        async with async_session() as db:
            query = select(Chunk)

            if request.skip_processed:
                subquery = select(Evidence.chunk_id).distinct()
                query = query.where(~Chunk.id.in_(subquery))

            if request.limit:
                query = query.limit(request.limit)

            result = await db.execute(query)
            chunks = result.scalars().all()
            total_chunks = len(chunks)

            update_job(job_id, total_items=total_chunks)

            if total_chunks == 0:
                update_job(
                    job_id,
                    status=JobStatus.COMPLETED,
                    completed_at=datetime.utcnow(),
                    progress=100.0,
                    result={"message": "No chunks to process"},
                )
                return

            llm_client = get_llm_client(request.provider)

            async with llm_client:
                service = ExtractionService(
                    db=db,
                    llm_client=llm_client,
                )

                def progress_callback(processed: int, total: int) -> None:
                    progress = (processed / total) * 100 if total > 0 else 0
                    update_job(
                        job_id,
                        processed_items=processed,
                        progress=progress,
                    )

                batch_result = await service.extract_all(
                    limit=request.limit,
                    skip_processed=request.skip_processed,
                    progress_callback=progress_callback,
                )

                update_job(
                    job_id,
                    status=JobStatus.COMPLETED,
                    completed_at=datetime.utcnow(),
                    progress=100.0,
                    error_count=batch_result.errors,
                    result={
                        "processed_chunks": batch_result.processed_chunks,
                        "total_entities": batch_result.total_entities,
                        "total_relations": batch_result.total_relations,
                        "new_entities": batch_result.new_entities,
                        "new_relations": batch_result.new_relations,
                        "errors": batch_result.errors,
                    },
                )

    except Exception as e:
        logger.error("Fallback KG build failed", job_id=str(job_id), error=str(e))
        update_job(
            job_id,
            status=JobStatus.FAILED,
            completed_at=datetime.utcnow(),
            error_message=str(e),
        )
    finally:
        await engine.dispose()


# =============================================================================
# Endpoints
# =============================================================================


@router.post(
    "/kg-build",
    response_model=JobCreateResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Trigger KG build job",
    description=(
        "Start an async job to extract entities and relations from document chunks. "
        "The job is dispatched to a Celery worker via Redis queue. "
        "If Celery is unavailable, falls back to in-process execution."
    ),
)
async def trigger_kg_build(
    request: KGBuildRequest,
    background_tasks: BackgroundTasks,
) -> JobCreateResponse:
    """Trigger async KG build job via Celery queue."""
    # Create job in shared Redis store
    job = create_job(JobType.KG_BUILD)
    job_id = job["id"]

    # Try dispatching to Celery worker
    if _celery_available():
        celery_task_id = _dispatch_to_celery(job_id, request)
        if celery_task_id:
            logger.info(
                "Job dispatched to Celery queue",
                job_id=str(job_id),
                celery_task_id=celery_task_id,
                provider=request.provider,
                limit=request.limit,
            )
            return JobCreateResponse(
                id=job_id,
                job_type=JobType.KG_BUILD,
                status=JobStatus.PENDING,
                message=(
                    "KG build job queued for Celery worker. "
                    "Use GET /jobs/{id} to check status."
                ),
            )

    # Fallback: run in-process via BackgroundTasks
    logger.info(
        "Celery unavailable, using BackgroundTasks fallback",
        job_id=str(job_id),
        provider=request.provider,
        limit=request.limit,
    )
    background_tasks.add_task(
        _fallback_kg_build_task,
        job_id,
        request,
        settings.db_url,
    )

    return JobCreateResponse(
        id=job_id,
        job_type=JobType.KG_BUILD,
        status=JobStatus.PENDING,
        message=(
            "KG build job started (in-process fallback). "
            "Use GET /jobs/{id} to check status."
        ),
    )


@router.get(
    "",
    response_model=PaginatedResponse[JobSummary],
    summary="List jobs",
    description="List all jobs with pagination.",
)
async def list_jobs_endpoint(
    page: int = Query(default=1, ge=1, description="Page number"),
    page_size: int = Query(default=20, ge=1, le=100, description="Items per page"),
    status_filter: JobStatus | None = Query(default=None, description="Filter by status"),
) -> PaginatedResponse[JobSummary]:
    """List jobs with pagination from Redis-backed store."""
    offset = (page - 1) * page_size
    jobs_page, total = list_jobs(
        status_filter=status_filter,
        offset=offset,
        limit=page_size,
    )

    items = [
        JobSummary(
            id=j["id"],
            job_type=j["job_type"],
            status=j["status"],
            progress=j["progress"],
            created_at=j["created_at"],
            completed_at=j.get("completed_at"),
        )
        for j in jobs_page
    ]

    return PaginatedResponse.create(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get(
    "/{job_id}",
    response_model=JobResponse,
    summary="Get job status",
    description="Get detailed status of a job including live progress from worker.",
)
async def get_job_status(job_id: UUID) -> JobResponse:
    """Get job status by ID from Redis-backed store."""
    job = get_job(job_id)

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found",
        )

    return JobResponse(**job)


@router.post(
    "/{job_id}/cancel",
    response_model=JobResponse,
    summary="Cancel a job",
    description="Cancel a pending or running job.",
)
async def cancel_job(job_id: UUID) -> JobResponse:
    """Cancel a job."""
    job = get_job(job_id)

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found",
        )

    if job["status"] in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Job {job_id} is already {job['status'].value}",
        )

    # Try to revoke the Celery task
    try:
        from src.workers.celery_app import celery_app
        celery_app.control.revoke(str(job_id), terminate=True)
    except Exception:
        pass  # Best-effort cancellation

    update_job(
        job_id,
        status=JobStatus.CANCELLED,
        completed_at=datetime.utcnow(),
    )

    logger.info("Job cancelled", job_id=str(job_id))

    return JobResponse(**get_job(job_id))


@router.get(
    "/progress/current",
    response_model=ExtractionProgress,
    summary="Get current extraction progress",
    description="Get real-time statistics about extraction progress from database.",
)
async def get_extraction_progress(
    db: AsyncSession = Depends(get_db),
) -> ExtractionProgress:
    """Get current extraction progress from database."""
    # Get total chunks
    total_result = await db.execute(select(func.count(Chunk.id)))
    total_chunks = total_result.scalar() or 0

    # Get processed chunks
    processed_result = await db.execute(
        select(func.count(func.distinct(Evidence.chunk_id)))
    )
    processed_chunks = processed_result.scalar() or 0

    # Get entity count
    entity_result = await db.execute(select(func.count(Entity.id)))
    entity_count = entity_result.scalar() or 0

    # Get relation count
    relation_result = await db.execute(select(func.count(Relation.id)))
    relation_count = relation_result.scalar() or 0

    # Count errors from active jobs
    _, total_jobs = list_jobs(status_filter=JobStatus.RUNNING)
    running_jobs, _ = list_jobs(status_filter=JobStatus.RUNNING)
    error_count = sum(j.get("error_count", 0) for j in running_jobs)

    return ExtractionProgress(
        chunks_total=total_chunks,
        chunks_processed=processed_chunks,
        chunks_remaining=total_chunks - processed_chunks,
        entities_extracted=entity_count,
        relations_extracted=relation_count,
        errors=error_count,
    )
