"""
Durable background task helpers.

Job state is persisted in the primary database so status survives process restarts
and can be shared across multiple API workers.
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Optional

from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import async_session
from app.models import TrackedJob
from app.ml.trainer import train_from_database

logger = logging.getLogger(__name__)


def _job_to_dict(job: TrackedJob) -> dict[str, Any]:
    """Serialize a tracked job row into API-friendly data."""
    metadata = json.loads(job.metadata_json) if job.metadata_json else {}
    result = json.loads(job.result_json) if job.result_json else None
    return {
        "id": job.id,
        "name": job.name,
        "status": job.status,
        "metadata": metadata,
        "created_at": job.created_at.isoformat() if job.created_at else None,
        "updated_at": job.updated_at.isoformat() if job.updated_at else None,
        "result": result,
        "error": job.error,
    }


async def get_all_jobs(db: AsyncSession) -> list[dict[str, Any]]:
    """Return tracked jobs ordered by recency."""
    result = await db.execute(select(TrackedJob).order_by(desc(TrackedJob.created_at)))
    return [_job_to_dict(job) for job in result.scalars().all()]


async def get_job_status(db: AsyncSession, job_id: str) -> Optional[dict[str, Any]]:
    """Return a tracked job record."""
    job = await db.get(TrackedJob, job_id)
    if not job:
        return None
    return _job_to_dict(job)


async def has_active_job(db: AsyncSession, name: str) -> bool:
    """Return whether a named job is currently queued or running."""
    result = await db.execute(
        select(TrackedJob.id).where(
            TrackedJob.name == name,
            TrackedJob.status.in_(("queued", "running")),
        ).limit(1)
    )
    return result.scalar_one_or_none() is not None


async def start_tracked_job(
    db: AsyncSession,
    name: str,
    metadata: Optional[dict[str, Any]] = None,
) -> str:
    """Create a tracked job record and return its id."""
    now = datetime.now(timezone.utc)
    job_id = f"job-{int(now.timestamp() * 1000)}"
    job = TrackedJob(
        id=job_id,
        name=name,
        status="queued",
        metadata_json=json.dumps(metadata or {}),
        created_at=now,
        updated_at=now,
    )
    db.add(job)
    await db.commit()
    return job_id


async def run_tracked_job(
    db: AsyncSession,
    job_id: str,
    work: Callable[[], Awaitable[Any]],
) -> None:
    """Run async work while updating durable job status."""
    job = await db.get(TrackedJob, job_id)
    if not job:
        raise KeyError(job_id)

    job.status = "running"
    job.updated_at = datetime.now(timezone.utc)
    await db.commit()

    try:
        result = await work()
        job.status = "completed"
        job.result_json = json.dumps(result, default=str)
        job.error = None
    except Exception as exc:
        job.status = "failed"
        job.error = str(exc)
        logger.exception("Tracked background job failed")
    finally:
        job.updated_at = datetime.now(timezone.utc)
        await db.commit()


def enqueue_training(n_trials: int = 50, run_comparison: bool = True):
    """Archived queue entrypoint kept for compatibility."""
    raise RuntimeError("External queue integration has been replaced by durable DB-backed tracked jobs.")


def enqueue_data_refresh(drugs: int = 5000, interactions: int = 100000):
    """Data refresh is not implemented."""
    raise NotImplementedError("Data refresh scripts have been archived. Use Archives/Scripts/ if needed.")


def get_job(job_id: str):
    """Legacy queue lookup is no longer supported."""
    return None


def train_models_job(n_trials: int, run_comparison: bool):
    """Run model training in a sync wrapper for compatibility."""
    async def _run():
        async with async_session() as db:
            await train_from_database(
                db_session=db,
                model_dir="./models",
                n_trials=n_trials,
                run_comparison=run_comparison,
            )

    asyncio.run(_run())
