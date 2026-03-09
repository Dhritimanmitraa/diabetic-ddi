"""
Model Retraining Scheduler.

Uses APScheduler to periodically check if new interaction data has
arrived. When sufficient new data accumulates, it triggers a training
run via the same pipeline used by the /ml/train endpoint.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from sqlalchemy import select, func

from app.database import async_session
from app.models import DrugInteraction

logger = logging.getLogger(__name__)

# ── Configuration (via env vars) ────────────────────────────────────────
RETRAIN_CHECK_HOURS = int(os.getenv("RETRAIN_CHECK_HOURS", "24"))
RETRAIN_MIN_NEW_ROWS = int(os.getenv("RETRAIN_MIN_NEW_ROWS", "500"))
RETRAIN_N_TRIALS = int(os.getenv("RETRAIN_N_TRIALS", "30"))

_scheduler: AsyncIOScheduler | None = None
_last_row_count: int | None = None


async def _check_and_retrain() -> None:
    """
    Compare the current interaction count against the last-known count.
    If enough new rows have appeared, kick off a training run.
    """
    global _last_row_count

    try:
        async with async_session() as session:
            result = await session.execute(
                select(func.count(DrugInteraction.id))
            )
            current_count = result.scalar() or 0

        if _last_row_count is None:
            # First run — just record the baseline.
            _last_row_count = current_count
            logger.info(
                "Retraining scheduler baseline set",
                extra={"interaction_count": current_count},
            )
            return

        new_rows = current_count - _last_row_count
        if new_rows >= RETRAIN_MIN_NEW_ROWS:
            logger.info(
                "Retraining triggered",
                extra={"new_rows": new_rows, "threshold": RETRAIN_MIN_NEW_ROWS},
            )
            await _do_retrain()
            _last_row_count = current_count
        else:
            logger.info(
                "Not enough new data for retraining",
                extra={"new_rows": new_rows, "threshold": RETRAIN_MIN_NEW_ROWS},
            )
    except Exception:
        logger.exception("Retraining check failed")


async def _do_retrain() -> None:
    """Run the actual training pipeline (same as /ml/train)."""
    from app.ml.trainer import train_from_database

    async with async_session() as session:
        summary = await train_from_database(
            session,
            model_dir="./models",
            n_trials=RETRAIN_N_TRIALS,
            run_comparison=False,
        )
        logger.info("Scheduled retraining finished", extra={"summary": str(summary)})

    # Bust the cached predictor so new requests use the fresh models.
    import app.ml.predictor as pred_mod
    pred_mod._predictor = None


def start_scheduler() -> None:
    """Create and start the APScheduler instance."""
    global _scheduler
    if _scheduler is not None:
        return

    _scheduler = AsyncIOScheduler()
    _scheduler.add_job(
        _check_and_retrain,
        trigger=IntervalTrigger(hours=RETRAIN_CHECK_HOURS),
        id="retrain_check",
        name="Periodic model retrain check",
        replace_existing=True,
    )
    _scheduler.start()
    logger.info(
        "Retraining scheduler started",
        extra={"interval_hours": RETRAIN_CHECK_HOURS, "min_new_rows": RETRAIN_MIN_NEW_ROWS},
    )


def stop_scheduler() -> None:
    """Gracefully shut down the scheduler."""
    global _scheduler
    if _scheduler is not None:
        _scheduler.shutdown(wait=False)
        _scheduler = None
        logger.info("Retraining scheduler stopped")
