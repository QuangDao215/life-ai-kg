"""
Database session management for FastAPI and standalone usage.

This module provides:
- FastAPI dependency for request-scoped database sessions
- Context manager for scripts and background workers
- Transaction helpers

Usage in FastAPI:
    @router.get("/entities")
    async def list_entities(db: AsyncSession = Depends(get_db)):
        result = await db.execute(select(Entity))
        return result.scalars().all()

Usage in scripts/workers:
    async with get_db_context() as db:
        result = await db.execute(select(Entity))
        entities = result.scalars().all()
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncSession

from src.db.base import AsyncSessionLocal


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency that provides a database session.

    Creates a new session for each request and ensure it's properly closed when the rquest completes.

    The session is NOT auto-committed - you must explicitly call 'await db.commit()' to persist changes.

    Usage:
        @router.get("/entities")
        async def create_entity(
            entity: EntityCreate,
            db: AsyncSession = Depends(get_db)
        ):
            db_entity = Entity(**entity.model_dump())
            db.add(db_entity)
            await db.commit()
            await db.refresh(db_entity)
            return db_entity

    Error handling:
        If an exception occurs, the session is rolled back automatically
        when it's closed (due to expire_on_commit=False in factory).
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


@asynccontextmanager
async def get_db_context() -> AsyncGenerator[AsyncSession, None]:
    """
    Context manager for database sessions outside of FastAPI.

    Use this in:
    - CLI scripts (fetch_pubmed.py, eval.py)
    - Celery workers
    - Tests
    - Any sync code that needs database access

    Usage:
        async with get_db_context() as db:
            entity = Entity(name="Ambroxol", type=EntityType.COMPOUND_DRUG)
            db.add(entity)
            await db.commit()

    The session is automatically closed when exiting the context, even if an exception occurs.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


@asynccontextmanager
async def transaction(session: AsyncSession) -> AsyncGenerator[AsyncSession, None]:
    """
    Context manager for explicit transaction control.

    Wraps operations in a transaction that:
    - Commits on successful completion
    - Rolls back on any exception

    Usage:
        async with get_db_context() as db:
            async with transaction(db):
                db.add(entity1)
                db.add(entity2)
                # Both are committed or neither is committed

    Note: SQLAlchemy sessions already have implicit transactions, but this makes
    the boundary explicit handles commit/rollback.
    """
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
