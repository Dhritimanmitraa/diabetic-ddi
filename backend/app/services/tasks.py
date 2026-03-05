"""
Background task helpers using RQ.

Note: Scripts folder has been archived. These functions are placeholders
for when background training/data refresh is needed.
"""
import os
import asyncio
from typing import Optional

try:
    import redis
    from rq import Queue
    from rq.job import Job
    RQ_AVAILABLE = True
except ImportError:
    RQ_AVAILABLE = False
    Queue = None
    Job = None

from app.database import async_session
from app.ml.trainer import train_from_database

from app.config import get_settings as _get_settings
REDIS_URL = _get_settings().REDIS_URL


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
