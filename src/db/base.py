"""
SQLAlchemy Base Configuration and Mixins.

This module provides:
- Async engine and session factory configuration
- Base declarative class for all models
- Reusable mixins (UUID7 primary key, timestamps)
- Type-annotated column helpers

All models in this project should inherit from `Base` and use the provided
mixins for consistency.
"""

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, MetaData, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.asyncio import AsyncAttrs, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, declared_attr, mapped_column
from uuid6 import uuid7

from src.core.config import settings

# =============================================================================
# DATABASE ENGINE CONFIGURATION
# =============================================================================

# Naming convention for database constraints
# This ensures consistent, predictable names for indexes, foreign keys, etc.
# Critical for Alembic migrations to work correctly across environments
NAMING_CONVENTION: dict[str, str] = {
    "ix": "ix_%(column_0_label)s",                    # Index
    "uq": "uq_%(table_name)s_%(column_0_name)s",      # Unique constraint
    "ck": "ck_%(table_name)s_%(constraint_name)s",    # Check constraint
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",  # Foreign key
    "pk": "pk_%(table_name)s",                        # Primary key
}

# MetaData instance with naming convention applied
# This is passed to the Base class so all models use consistent naming
metadata = MetaData(naming_convention=NAMING_CONVENTION)

# Async database engine
# - pool_pre_ping: Validates connections before use (handles stale connections)
# - echo: Logs SQL statements when in development mode
# - pool_size: Number of persistent connections in the pool
# - max_overflow: Additional connections allowed beyond pool_size
engine = create_async_engine(
    settings.db_url,
    echo=settings.is_development,  # Log SQL in dev mode only
    pool_pre_ping=True,            # Check connection health before use
    pool_size=5,                   # Base number of connections
    max_overflow=10,               # Allow up to 15 total (5 + 10)
)

# Async session factory
# - expire_on_commit=False: Objects remain accessible after commit
#   (important for returning data from async endpoints)
# - autoflush=False: Don't auto-flush before queries
#   (gives explicit control over when writes happen)
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    expire_on_commit=False,
    autoflush=False,
)


# =============================================================================
# BASE CLASS
# =============================================================================

class Base(AsyncAttrs, DeclarativeBase):
    """
    Base class for all SQLAlchemy models.

    Features:
    - AsyncAttrs: Enables `await` on lazy-loaded relationships
    - DeclarativeBase: Modern SQLAlchemy 2.0 declarative base
    - Custom metadata with naming conventions
    - Automatic __tablename__ generation from class name

    Example:
        class User(Base):
            # __tablename__ automatically set to "users"
            name: Mapped[str] = mapped_column(String(100))
    """

    # Apply naming convention to all models
    metadata = metadata

    # Type annotation for registry (helps with IDE support)
    __name__: str

    @declared_attr.directive
    def __tablename__(cls) -> str:
        """
        Automatically generate table name from class name.

        Converts CamelCase to snake_case and pluralizes:
        - Entity -> entities
        - AuditLog -> audit_logs
        - Document -> documents

        This follows database naming conventions and avoids manually
        specifying __tablename__ on each model.
        """
        # Convert CamelCase to snake_case
        name = cls.__name__
        # Insert underscore before uppercase letters (except first)
        snake_case = "".join(
            f"_{char.lower()}" if char.isupper() and i > 0 else char.lower()
            for i, char in enumerate(name)
        )
        # Simple pluralization (add 's' or 'es')
        if snake_case.endswith("y"):
            return snake_case[:-1] + "ies"      # Entity -> entities
        elif snake_case.endswith("s"):
            return snake_case + "es"            # Class -> classes
        else:
            return snake_case + "s"             # Document -> documents

    def to_dict(self) -> dict[str, Any]:
        """
        Convert model instance to dictionary.

        Useful for:
        - Audit logging (serializing old_data before deletion)
        - Debugging
        - Simple serialization (though Pydantic schemas are preferred for API)

        Handle special types:
        - UUID -> string
        - datetime -> ISO format string
        - Enum -> value
        """
        result = {}
        for column in self.__table__.columns:
            value = getattr(self, column.name)
            # Handle special types for JSON serialization
            if isinstance(value, uuid.UUID):
                value = str(value)
            elif isinstance(value, datetime):
                value = value.isoformat()
            elif hasattr(value, "value"):   # Enum
                value = value.value
            result[column.name] = value
        return result


# =============================================================================
# MIXINS
# =============================================================================

class UUIDMixin:
    """
    Mixin that provides a UUID7 primary key.

    Usage:
        class MyModel(UUIDMixin, Base):
            name: Mapped[str]

    The 'id' column will be:
    - Primary key
    - Auto-generated on insert
    - PostgreSQL native UUID type
    """

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid7,
        sort_order=-100, # Ensure 'id' appears first in the table definition
    )


class TimestampMixin:
    """
    Mixin that provides created_at and updated_at timestamps.

    - created_at: Set once when row is inserted (server-side default)
    - updated_at: Updated automatically on every modification

    Usage:
        class MyModel(TimestampMixin, Base):
            name: Mapped[str]

    Note: updated_at uses 'onupdate=func.now()' which triggers on UPDATE.
    """

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        sort_order=100, # Appear near end of columns
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
        sort_order=101,
    )


class CreatedAtMixin:
    """
    Mixin that provides only created_at timestamp.

    Use this for immutable records where updated_at doesn't make sense:
    - Evidence records (never modified, only created)
    - Audit log entries (append-only)
    - Chunks (recreated, not updated)

    Usage:
        class Evidence(CreatedAtMixin, Base):
            quote: Mapped[str]
    """

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        sort_order=100,
    )


# =============================================================================
# COMBINED BASE CLASSES (Convenience)
# =============================================================================

class UUIDBase(UUIDMixin, Base):
    """
    Abstract base with UUID7 primary key.

    Use this when you need UUID but handle timestamps yourself.
        class CustomModel(UUIDBase):
            __abstract__ = False
            name: Mapped[str]

    """
    ___abstract__ = True


class UUIDTimestampBase(UUIDMixin, TimestampMixin, Base):
    """
    Abstract base with UUID7 + created_at + updated_at.

    Most common base class for mutable entities:
        class Entity(UUIDTimestampBase):
            __abstract__ = False
            name: Mapped[str]
    """
    ___abstract__ = True


class UUIDCreatedBase(UUIDMixin, CreatedAtMixin, Base):
    """
    Abstract base with UUID7 + created_at.

    For immutable/append-only records:
        class AuditLog(UUIDCreatedBase):
            __abstract__ = False
            action: Mapped[str]
    """
    __abstract__ = True


# =============================================================================
# DATABASE LIFECYCLE UTILITIES
# =============================================================================

async def init_db() -> None:
    """
    Initialize the database by creating all tables.

    Note: In production, use Alembic migrations instead.
    This is useful for testing and rapid development.

    Usage:
        async def startup():
            await init_db()
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def drop_db() -> None:
    """
    Drop all tables in the database.

    Warning: This is destructive! Only use in testing.

    Usage:
        @pytest.fixture
        async def clean_db():
            await drop_db()
            await init_db()
            yield
            await drop_db()
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


async def dispose_engine() -> None:
    """
    Dispose of the engine and close all connections.

    Call this during application shutdown to cleanly release database connections.

    Usage:
        async def shutdown():
            await dispose_engine()
    """
    await engine.dispose()
