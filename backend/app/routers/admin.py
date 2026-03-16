"""Protected admin endpoints for system and job status."""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.database import get_db
from app.models import ComparisonLog, Drug, DrugInteraction, MLPrediction
from app.services.api_client import get_api_client
from app.services.auth import require_api_key
from app.services.cache import get_redis_client
from app.services.gemini_client import get_gemini_client
from app.services.tasks import get_all_jobs

router = APIRouter(prefix="/admin", tags=["Admin"], dependencies=[Depends(require_api_key)])


@router.get("/system-status")
async def system_status(db: AsyncSession = Depends(get_db)):
    """Return deploy-oriented system status for operators."""
    settings = get_settings()
    gemini = get_gemini_client()
    redis_client = await get_redis_client()
    api_client = get_api_client()

    total_drugs = await db.scalar(select(func.count(Drug.id))) or 0
    total_interactions = await db.scalar(select(func.count(DrugInteraction.id))) or 0
    total_comparisons = await db.scalar(select(func.count(ComparisonLog.id))) or 0
    total_predictions = await db.scalar(select(func.count(MLPrediction.id))) or 0

    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "app": {
            "name": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "debug": settings.DEBUG,
        },
        "security": {
            "api_key_configured": bool(settings.API_KEY),
            "rate_limit_per_minute": settings.RATE_LIMIT_REQUESTS_PER_MIN,
        },
        "services": {
            "redis": redis_client is not None,
            "gemini": {
                "available": gemini.is_available,
                "sdk": gemini.sdk,
                "model": gemini.model,
            },
            "external_apis": api_client.get_health_status(),
        },
        "data": {
            "total_drugs": total_drugs,
            "total_interactions": total_interactions,
            "total_comparisons": total_comparisons,
            "total_ml_predictions": total_predictions,
        },
        "jobs": get_all_jobs(),
    }


@router.get("/jobs")
async def list_jobs():
    """Return tracked background jobs."""
    return {"jobs": get_all_jobs()}