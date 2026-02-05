"""
Celery tasks for KG extraction pipeline.

This module defines the async-compatible Celery tasks that run in
the worker process. Each task bridges Celery's synchronous execution
model with our async extraction code using asyncio.run().

Usage:
    # From API (dispatch to queue):
    from src.workers.tasks import kg_build_task
    kg_build_task.delay(str(job_id), provider="gemini", limit=50)

    # Start worker:
    celery -A src.workers.celery_app worker -l info -P solo

Requirements addressed:
- Assignment: "KG build pipeline must run asynchronously using a queue/worker architecture"
- REQ-KG-06: Job status tracking with progress and error details
"""

import asyncio
from datetime import datetime
from uuid import UUID

from src.core.logging import get_logger
from src.workers.celery_app import celery_app
from src.workers.job_store import update_job

logger = get_logger(__name__)


async def _run_kg_extraction(
    job_id: UUID,
    provider: str,
    limit: int | None,
    skip_processed: bool,
    db_url: str,
) -> dict:
    """
    Async implementation of KG extraction.

    Runs entity and relation extraction on unprocessed chunks,
    updating job progress along the way.
    """
    from sqlalchemy import select
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    from src.db.models import Chunk, Evidence
    from src.schemas.jobs import JobStatus
    from src.services.extraction import ExtractionService
    from src.services.llm_client import get_llm_client

    engine = create_async_engine(db_url, echo=False)
    async_session = async_sessionmaker(engine, expire_on_commit=False)

    try:
        update_job(job_id, status=JobStatus.RUNNING, started_at=datetime.utcnow())

        async with async_session() as db:
            # Get chunks to process
            query = select(Chunk)

            if skip_processed:
                subquery = select(Evidence.chunk_id).distinct()
                query = query.where(~Chunk.id.in_(subquery))

            if limit:
                query = query.limit(limit)

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
                return {"processed_chunks": 0, "message": "No chunks to process"}

            # Initialize LLM client and extraction service
            llm_client = get_llm_client(provider)

            async with llm_client:
                service = ExtractionService(
                    db=db,
                    llm_client=llm_client,
                )

                # Progress callback — updates Redis job store
                def progress_callback(processed: int, total: int) -> None:
                    progress = (processed / total) * 100 if total > 0 else 0
                    update_job(
                        job_id,
                        processed_items=processed,
                        progress=progress,
                    )

                # Run extraction
                batch_result = await service.extract_all(
                    limit=limit,
                    skip_processed=skip_processed,
                    progress_callback=progress_callback,
                )

                result_data = {
                    "processed_chunks": batch_result.processed_chunks,
                    "total_entities": batch_result.total_entities,
                    "total_relations": batch_result.total_relations,
                    "new_entities": batch_result.new_entities,
                    "new_relations": batch_result.new_relations,
                    "errors": batch_result.errors,
                }

                update_job(
                    job_id,
                    status=JobStatus.COMPLETED,
                    completed_at=datetime.utcnow(),
                    progress=100.0,
                    error_count=batch_result.errors,
                    result=result_data,
                )

                return result_data

    except Exception as e:
        logger.error("KG extraction failed", job_id=str(job_id), error=str(e))
        update_job(
            job_id,
            status=JobStatus.FAILED,
            completed_at=datetime.utcnow(),
            error_message=str(e),
        )
        raise
    finally:
        await engine.dispose()


@celery_app.task(
    name="src.workers.tasks.kg_build_task",
    bind=True,
    max_retries=1,
    acks_late=True,
)
def kg_build_task(
    self,
    job_id_str: str,
    provider: str = "gemini",
    limit: int | None = None,
    skip_processed: bool = True,
) -> dict:
    """
    Celery task: Build Knowledge Graph from document chunks.

    This task runs in the Celery worker process and bridges to
    the async extraction pipeline using asyncio.run().

    Args:
        job_id_str: Job UUID as string (Celery requires JSON-serializable args)
        provider: LLM provider name (gemini, mock)
        limit: Max chunks to process
        skip_processed: Skip already-processed chunks

    Returns:
        dict with extraction results
    """
    from src.core.config import settings

    job_id = UUID(job_id_str)

    logger.info(
        "Celery worker: starting KG build",
        job_id=job_id_str,
        provider=provider,
        limit=limit,
    )

    # Bridge async code into Celery's sync execution
    result = asyncio.run(
        _run_kg_extraction(
            job_id=job_id,
            provider=provider,
            limit=limit,
            skip_processed=skip_processed,
            db_url=settings.db_url,
        )
    )

    logger.info(
        "Celery worker: KG build complete",
        job_id=job_id_str,
        result=result,
    )

    return result
