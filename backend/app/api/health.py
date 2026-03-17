"""Health and statistics endpoints."""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.database import get_db
from app.models import Drug, DrugInteraction
from app.schemas import DatabaseStats
from app.services.cache import cache_db_stats, get_cached_db_stats

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter()


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _model_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "models"


@router.get("/", tags=["Health"])
async def root():
    """Root endpoint - API health check."""
    return {
        "message": "Drug-Drug Interaction Prediction API",
        "version": settings.APP_VERSION,
        "status": "healthy",
        "docs": "/docs",
    }


@router.get("/health", tags=["Health"])
async def health_check():
    """Detailed health check."""
    return {
        "status": "healthy",
        "timestamp": _utc_now().isoformat(),
        "version": settings.APP_VERSION,
    }


@router.get("/health/live", tags=["Health"])
async def liveness():
    """Simple liveness probe."""
    return {"status": "alive"}


@router.get("/health/ready", tags=["Health"])
async def readiness(db: AsyncSession = Depends(get_db)):
    """Readiness probe checking database connectivity and local model artifacts."""
    await db.execute(select(func.count(Drug.id)))

    model_dir = _model_dir()
    has_models = all(
        os.path.exists(model_dir / filename)
        for filename in (
            "feature_extractor.pkl",
            "random_forest_model.pkl",
            "xgboost_model.pkl",
            "lightgbm_model.pkl",
        )
    )

    return {
        "status": "ready" if has_models else "degraded",
        "models_loaded": has_models,
        "request_tracing": True,
        "admin_api_protected": bool(settings.API_KEY),
    }


@router.get("/health/apis", tags=["Health"])
async def api_health_status():
    """Get health status of external APIs and LLM configuration."""
    try:
        from app.services.api_client import get_api_client

        client = get_api_client()
        status = client.get_health_status()

        return {
            "status": "healthy",
            "external_apis": status,
            "llm_services": {
                "ollama_configured": bool(settings.OLLAMA_HOST),
                "gemini_configured": bool(settings.GOOGLE_API_KEY),
                "fallback_to_templates": settings.LLM_FALLBACK_TO_TEMPLATES,
            },
            "api_keys": {
                "openfda_key_configured": bool(settings.OPENFDA_API_KEY),
                "umls_key_configured": bool(settings.UMLS_API_KEY),
            },
            "timestamp": _utc_now().isoformat(),
        }
    except Exception as exc:
        logger.error(f"API health check error: {exc}")
        return {
            "status": "error",
            "error": str(exc),
            "timestamp": _utc_now().isoformat(),
        }


@router.get("/stats", response_model=DatabaseStats, tags=["Statistics"])
async def get_statistics(db: AsyncSession = Depends(get_db)):
    """Get cached database statistics."""
    cached_stats = await get_cached_db_stats()
    if cached_stats is not None:
        logger.debug("Cache HIT for database stats")
        return DatabaseStats(**cached_stats)

    logger.debug("Cache MISS for database stats")

    total_drugs = (await db.execute(select(func.count(Drug.id)))).scalar()
    total_interactions = (await db.execute(select(func.count(DrugInteraction.id)))).scalar()

    severity_results = (
        await db.execute(
            select(DrugInteraction.severity, func.count(DrugInteraction.id)).group_by(
                DrugInteraction.severity
            )
        )
    ).all()
    severity_counts = {row[0]: row[1] for row in severity_results if row[0]}
    last_updated = _utc_now()

    stats_data = {
        "total_drugs": total_drugs,
        "total_interactions": total_interactions,
        "interactions_by_severity": severity_counts,
        "last_updated": last_updated.isoformat(),
    }
    await cache_db_stats(stats_data)

    return DatabaseStats(
        total_drugs=total_drugs,
        total_interactions=total_interactions,
        interactions_by_severity=severity_counts,
        last_updated=last_updated,
    )
