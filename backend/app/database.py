"""Database configuration and session management."""
import asyncio
import logging
import os

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase

from app.config import get_settings

settings = get_settings()
_db_init_lock = asyncio.Lock()
_db_initialized = False
logger = logging.getLogger(__name__)


def _engine_kwargs() -> dict:
    """Build engine options appropriate for the configured driver."""
    kwargs = {
        "echo": settings.DEBUG,
        "future": True,
    }
    database_url = settings.DATABASE_URL.lower()
    if database_url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
        return kwargs

    kwargs.update(
        {
            "pool_pre_ping": True,
            "pool_size": settings.DATABASE_POOL_SIZE,
            "max_overflow": settings.DATABASE_MAX_OVERFLOW,
            "pool_recycle": settings.DATABASE_POOL_RECYCLE_SECONDS,
        }
    )
    return kwargs

# Create async engine
engine = create_async_engine(
    settings.DATABASE_URL,
    **_engine_kwargs(),
)

# Session factory
async_session = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)


class Base(DeclarativeBase):
    """Base class for all models."""
    pass


async def get_db() -> AsyncSession:
    """Dependency to get database session."""
    global _db_initialized
    if not _db_initialized:
        async with _db_init_lock:
            if not _db_initialized:
                await init_db()

    async with async_session() as session:
        try:
            yield session
        finally:
            await session.close()


async def init_db():
    """Load model metadata and optionally create tables for local development."""
    global _db_initialized
    if _db_initialized:
        return

    # Import all models to ensure they're registered with Base.metadata
    from app import models  # noqa: F401
    from app.diabetic import models as diabetic_models  # noqa: F401
    from app.prescription import models as prescription_models  # noqa: F401

    should_create_tables = settings.DB_AUTO_CREATE or bool(os.getenv("PYTEST_CURRENT_TEST"))
    if should_create_tables:
        logger.warning(
            "Creating database tables from ORM metadata. Use Alembic migrations for production deployments."
        )
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    else:
        logger.info("Skipping ORM create_all; expecting schema to be managed by Alembic")

    _db_initialized = True
