"""
ML Training & Status Router.

Provides admin endpoints for triggering model re-training and
inspecting the current model status.
"""
from __future__ import annotations

import json
import logging
import os

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.database import get_db
from app.models import Drug
from app.services.auth import require_api_key
from app.services.rate_limiter import rate_limit_dependency
from app.services.tasks import get_job_status, has_active_job, run_tracked_job, start_tracked_job

router = APIRouter(prefix="/ml", tags=["ML"])
logger = logging.getLogger(__name__)
settings = get_settings()


class TrainRequest(BaseModel):
    n_trials: int = Field(default=50, ge=5, le=500, description="Optuna trials per model")
    run_comparison: bool = Field(default=False, description="Compare Bayesian / Grid / Random")


class TrainResponse(BaseModel):
    status: str
    message: str


class ModelStatusResponse(BaseModel):
    models_loaded: list[str] = []
    optimal_threshold: float = 0.5
    threshold_method: str = "default"
    ml_available: bool = False
    model_metrics: dict = {}
    scheduler_running: bool = False
    retrain_interval_hours: int = 0
    retrain_min_new_rows: int = 0


class PredictRequest(BaseModel):
    drug1_name: str = Field(..., min_length=1)
    drug2_name: str = Field(..., min_length=1)


@router.post("/train", response_model=TrainResponse)
async def trigger_training(
    body: TrainRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    _: None = Depends(require_api_key),
    __: None = rate_limit_dependency(limit=settings.HEAVY_RATE_LIMIT_REQUESTS_PER_MIN, key_prefix="ml_train"),
):
    """Trigger asynchronous DDI model training."""
    if await has_active_job(db, "ml_training"):
        raise HTTPException(status_code=409, detail="Training already in progress")

    job_id = await start_tracked_job(
        db,
        "ml_training",
        {"n_trials": body.n_trials, "run_comparison": body.run_comparison},
    )
    background_tasks.add_task(_run_training, job_id, body.n_trials, body.run_comparison)
    return TrainResponse(status="started", message=f"Training started in background (job {job_id})")


@router.get("/jobs/{job_id}")
async def training_job_status(
    job_id: str,
    db: AsyncSession = Depends(get_db),
    _: None = Depends(require_api_key),
):
    """Return status for a tracked ML background job."""
    job = await get_job_status(db, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.get("/status", response_model=ModelStatusResponse)
async def model_status():
    """Return information about the currently-loaded ML models and scheduler."""
    from app.ml.scheduler import RETRAIN_CHECK_HOURS, RETRAIN_MIN_NEW_ROWS, _scheduler

    resp = ModelStatusResponse(
        scheduler_running=_scheduler is not None and _scheduler.running,
        retrain_interval_hours=RETRAIN_CHECK_HOURS,
        retrain_min_new_rows=RETRAIN_MIN_NEW_ROWS,
    )
    try:
        from app.ml.predictor import get_predictor

        predictor = get_predictor("./models")
        if predictor.is_loaded:
            info = predictor.get_model_info()
            resp.models_loaded = info.get("models_loaded", [])
            resp.optimal_threshold = info.get("optimal_threshold", 0.5)
            resp.threshold_method = info.get("threshold_method", "default")
            resp.ml_available = True
            resp.model_metrics = info.get("model_metrics", {})
    except Exception as e:
        logger.warning("Could not load predictor for status check: %s", e)

    return resp


@router.post("/predict")
async def predict_interaction(body: PredictRequest, db: AsyncSession = Depends(get_db)):
    """Predict drug-drug interaction using ML ensemble."""
    from app.ml.predictor import get_predictor

    predictor = get_predictor("./models")
    if not predictor.is_loaded:
        raise HTTPException(status_code=503, detail="ML models not loaded. Train models first.")

    drug1_dict = await _drug_to_dict(db, body.drug1_name)
    drug2_dict = await _drug_to_dict(db, body.drug2_name)

    try:
        result = predictor.predict(drug1_dict, drug2_dict)
        return result.to_dict()
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/model-info")
async def model_info():
    """Return detailed model info, metrics, and feature importance for the dashboard."""
    from app.ml.predictor import get_predictor

    predictor = get_predictor("./models")
    if not predictor.is_loaded:
        return {"status": "not_loaded"}

    info = predictor.get_model_info()
    feature_importance = predictor.get_feature_importance()

    return {
        "status": "loaded",
        "models": info,
        "feature_importance": feature_importance,
    }


@router.get("/comparison")
async def optimization_comparison():
    """Return Bayesian vs Grid vs Random optimization comparison results."""
    model_dir = "./models"
    comparisons = []
    bayesian_wins = 0

    for name in ["random_forest", "xgboost", "lightgbm"]:
        path = os.path.join(model_dir, f"{name}_comparison.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            comparisons.append(data)

    if not comparisons:
        return {"status": "not_available"}

    detailed = []
    for comp in comparisons:
        summary = comp.get("comparison_summary", {})
        efficiency = summary.get("efficiency_gain", {})

        is_best = summary.get("winner", "") == "bayesian"
        if is_best:
            bayesian_wins += 1

        detailed.append(
            {
                "model": summary.get("model_type", "unknown"),
                "comparison": comp,
                "bayesian_is_best": is_best,
                "trial_reduction_percent": efficiency.get("trial_reduction_percent", 0),
                "time_reduction_percent": efficiency.get("time_reduction_percent", 0),
            }
        )

    total = len(detailed)
    avg_reduction = sum(d["trial_reduction_percent"] for d in detailed) / total if total else 0

    return {
        "status": "loaded",
        "bayesian_wins": bayesian_wins,
        "total_models_compared": total,
        "average_trial_reduction_percent": avg_reduction,
        "detailed_comparisons": detailed,
    }


async def _drug_to_dict(db: AsyncSession, name: str) -> dict:
    """Look up a drug by name and return a feature dict."""
    result = await db.execute(select(Drug).where(Drug.name.ilike(name)).limit(1))
    drug = result.scalars().first()
    if drug:
        return {
            "name": drug.name,
            "generic_name": drug.generic_name,
            "drug_class": drug.drug_class,
            "description": drug.description,
            "mechanism": drug.mechanism,
            "indication": drug.indication,
            "molecular_weight": drug.molecular_weight,
            "matched": True,
        }
    return {"name": name, "matched": False}


async def _run_training(job_id: str, n_trials: int, run_comparison: bool) -> None:
    """Run training and persist state transitions in the tracked job table."""
    from app.database import async_session

    async def _work():
        async with async_session() as session:
            from app.ml.trainer import train_from_database

            summary = await train_from_database(
                session,
                model_dir="./models",
                n_trials=n_trials,
                run_comparison=run_comparison,
            )
            logger.info("Training finished: %s", summary)

        import app.ml.predictor as pred_mod

        pred_mod._predictor = None
        return summary

    async with async_session() as session:
        await run_tracked_job(session, job_id, _work)
