"""
Background task helpers using RQ.
"""
import os
import asyncio
from typing import Optional

import redis
from rq import Queue
from rq.job import Job

from app.database import async_session
from app.ml.trainer import train_from_database
from scripts.fetch_real_data import fetch_and_load

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")


def get_queue() -> Queue:
    conn = redis.from_url(REDIS_URL)
    return Queue("default", connection=conn)


def enqueue_training(n_trials: int = 50, run_comparison: bool = True) -> Job:
    queue = get_queue()
    return queue.enqueue(train_models_job, n_trials, run_comparison, job_timeout="1h")


def enqueue_data_refresh(drugs: int = 5000, interactions: int = 100000) -> Job:
    queue = get_queue()
    return queue.enqueue(fetch_data_job, drugs, interactions, job_timeout="2h")


def get_job(job_id: str) -> Optional[Job]:
    queue = get_queue()
    return queue.fetch_job(job_id)


def train_models_job(n_trials: int, run_comparison: bool):
    """Run model training inside RQ (sync wrapper)."""
    async def _run():
        async with async_session() as db:
            await train_from_database(db_session=db, model_dir="./models", n_trials=n_trials, run_comparison=run_comparison)
    asyncio.run(_run())


def fetch_data_job(drugs: int, interactions: int):
    """Run data fetch inside RQ (sync wrapper)."""
    asyncio.run(fetch_and_load(target_drugs=drugs, target_interactions=interactions))

