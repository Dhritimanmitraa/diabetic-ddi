"""
Background task helpers using RQ.

Note: Scripts folder has been archived. These functions are placeholders
for when background training/data refresh is needed.
"""
import os
import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Optional

logger = logging.getLogger(__name__)

try:
    import redis
    from rq import Queue
    from rq.job import Job
    RQ_AVAILABLE = True
except Exception as exc:
    RQ_AVAILABLE = False
    Queue = None
    Job = None
    logger.warning("RQ unavailable, background queue features disabled: %s", exc)

from app.database import async_session
from app.ml.trainer import train_from_database

from app.config import get_settings as _get_settings
REDIS_URL = _get_settings().REDIS_URL

_jobs: dict[str, dict[str, Any]] = {}


def get_all_jobs() -> list[dict[str, Any]]:
    """Return tracked in-process jobs ordered by recency."""
    return sorted(_jobs.values(), key=lambda item: item["created_at"], reverse=True)


def get_job_status(job_id: str) -> Optional[dict[str, Any]]:
    """Return a tracked job record."""
    return _jobs.get(job_id)


def start_tracked_job(name: str, metadata: Optional[dict[str, Any]] = None) -> str:
    """Create a tracked job record and return its id."""
    job_id = f"job-{int(datetime.now(timezone.utc).timestamp() * 1000)}"
    _jobs[job_id] = {
        "id": job_id,
        "name": name,
        "status": "queued",
        "metadata": metadata or {},
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "result": None,
        "error": None,
    }
    return job_id


async def run_tracked_job(job_id: str, work: Callable[[], Awaitable[Any]]) -> None:
    """Run async work while updating job status for observability."""
    job = _jobs[job_id]
    job["status"] = "running"
    job["updated_at"] = datetime.now(timezone.utc).isoformat()
    try:
        result = await work()
        job["status"] = "completed"
        job["result"] = result
    except Exception as exc:
        job["status"] = "failed"
        job["error"] = str(exc)
        logger.exception("Tracked background job failed")
    finally:
        job["updated_at"] = datetime.now(timezone.utc).isoformat()


def get_queue():
    if not RQ_AVAILABLE:
        raise RuntimeError("RQ/Redis not available. Install redis and rq packages.")
    import redis as r
    from rq import Queue as Q
    conn = r.from_url(REDIS_URL)
    return Q("default", connection=conn)


def enqueue_training(n_trials: int = 50, run_comparison: bool = True):
    """Enqueue model training job (requires Redis)."""
    if not RQ_AVAILABLE:
        raise RuntimeError("RQ not available")
    queue = get_queue()
    return queue.enqueue(train_models_job, n_trials, run_comparison, job_timeout="1h")


def enqueue_data_refresh(drugs: int = 5000, interactions: int = 100000):
    """Enqueue data refresh job (requires Redis and scripts folder)."""
    raise NotImplementedError("Data refresh scripts have been archived. Use Archives/Scripts/ if needed.")


def get_job(job_id: str):
    """Get job status by ID."""
    if not RQ_AVAILABLE:
        return None
    queue = get_queue()
    return queue.fetch_job(job_id)


def train_models_job(n_trials: int, run_comparison: bool):
    """Run model training inside RQ (sync wrapper)."""
    async def _run():
        async with async_session() as db:
            await train_from_database(db_session=db, model_dir="./models", n_trials=n_trials, run_comparison=run_comparison)
    asyncio.run(_run())
