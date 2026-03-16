"""
Drug-Drug Interaction Prediction API

Main FastAPI application entry point.
"""
from fastapi import FastAPI, Depends, Request, APIRouter, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from contextlib import asynccontextmanager
import logging
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from app.config import get_settings
from app.database import init_db, get_db
from app.models import Drug, DrugInteraction
from app.schemas import DatabaseStats
from app.services.cache import get_cached_db_stats, cache_db_stats
import os

# Import Routers
from app.diabetic.router import router as diabetic_router
from app.prescription.router import router as prescription_router
from app.routers.drugs import router as drugs_router
from app.routers.interactions import router as interactions_router
from app.routers.ocr import router as ocr_router
from app.routers.history import router as history_router
from app.routers.ml_router import router as ml_router
from app.routers.adherence import router as adherence_router
from app.routers.admin import router as admin_router
from app.services.rate_limiter import rate_limit

# ── Structured JSON Logging ─────────────────────────────────────────────
from pythonjsonlogger import jsonlogger

_log_handler = logging.StreamHandler(sys.stdout)
_log_handler.setFormatter(
    jsonlogger.JsonFormatter(
        fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
        rename_fields={"asctime": "timestamp", "levelname": "level"},
    )
)
logging.root.handlers.clear()
logging.root.addHandler(_log_handler)
logging.root.setLevel(logging.INFO)

logger = logging.getLogger(__name__)

settings = get_settings()


def _validate_startup_configuration() -> None:
    """Optionally fail fast in production when required secrets are missing."""
    is_production = settings.APP_ENV.lower() in {"prod", "production"}
    if not is_production:
        return

    if not settings.STRICT_STARTUP_VALIDATION:
        logger.warning("Production strict startup validation is disabled")
        return

    missing: list[str] = []

    if settings.REQUIRE_API_KEY_FOR_ADMIN and not settings.API_KEY:
        missing.append("API_KEY")

    if settings.REQUIRE_GEMINI_KEY and not (settings.GEMINI_API_KEY or settings.GOOGLE_API_KEY):
        missing.append("GEMINI_API_KEY/GOOGLE_API_KEY")

    if settings.ENABLE_CLOUD_SPEECH and not settings.CLOUD_SPEECH_API_KEY:
        missing.append("CLOUD_SPEECH_API_KEY")

    if missing:
        logger.error(
            "Startup validation failed",
            extra={"missing_settings": missing, "app_env": settings.APP_ENV},
        )
        raise RuntimeError(f"Missing required production settings: {', '.join(missing)}")

async def seed_initial_data():
    """Placeholder for seeding data if needed."""
    pass

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    _validate_startup_configuration()
    logger.info("Initializing database...")
    await init_db()
    logger.info("Database initialized successfully!")
    
    # Seed initial data if empty
    await seed_initial_data()

    # Start model-retraining scheduler
    from app.ml.scheduler import start_scheduler, stop_scheduler
    start_scheduler()
    
    yield
    
    # Shutdown
    stop_scheduler()
    logger.info("Shutting down...")
    from app.services.api_client import close_api_client
    await close_api_client()


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="""
    ## Drug-Drug Interaction Prediction System
    
    A machine learning-powered system to check drug interactions and find safe alternatives.
    
    ### Features:
    - **Check Interactions**: Verify if two drugs are safe to use together
    - **Image Recognition**: Extract drug names from photos using OCR
    - **Alternative Suggestions**: Get safe alternative medications
    - **Comprehensive Database**: 100,000+ drug interactions
    
    ### Severity Levels:
    - 🟢 **Minor**: Generally safe, minimal effects
    - 🟡 **Moderate**: Use caution, monitor for effects
    - 🟠 **Major**: Significant interaction, consult healthcare provider
    - 🔴 **Contraindicated**: Do NOT use together
    """,
    lifespan=lifespan
)

# Log drug search/list requests visibly in terminal
@app.middleware("http")
async def log_drug_requests(request: Request, call_next):
    """
    Log all /drugs and /drugs/search calls to console for visibility.
    """
    path = request.url.path
    if path.startswith("/v1/drugs") or path.startswith("/drugs"):
        client_ip = request.client.host if request.client else "unknown"
        logger.info(
            "http_request",
            extra={
                "path": path,
                "method": request.method,
                "query": dict(request.query_params),
                "client_ip": client_ip,
            },
        )
    response = await call_next(request)
    return response


@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    """Attach request tracing, audit logging, and global rate limiting."""
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.request_id = request_id
    start = time.perf_counter()

    try:
        await rate_limit(
            request,
            limit=settings.RATE_LIMIT_REQUESTS_PER_MIN,
            window_seconds=60,
            key_prefix="global",
        )
        response = await call_next(request)
    except HTTPException as exc:
        response = JSONResponse(
            status_code=exc.status_code,
            content={
                "detail": exc.detail,
                "request_id": request_id,
            },
        )
    except Exception as exc:
        logger.exception("Unhandled request error", extra={"request_id": request_id, "path": request.url.path})
        response = JSONResponse(
            status_code=500,
            content={
                "detail": "Internal server error",
                "request_id": request_id,
            },
        )

    duration_ms = round((time.perf_counter() - start) * 1000, 2)
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-MS"] = str(duration_ms)

    logger.info(
        "request_completed",
        extra={
            "request_id": request_id,
            "path": request.url.path,
            "method": request.method,
            "status_code": response.status_code,
            "duration_ms": duration_ms,
            "client_ip": request.client.host if request.client else "unknown",
        },
    )
    return response

# CORS configuration — lock down to known origins
_allowed_origins = [
    origin.strip()
    for origin in settings.CORS_ORIGINS.split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-API-Key"],
)

# ── API v1 Router ────────────────────────────────────────────────────────
v1 = APIRouter(prefix="/v1")
v1.include_router(diabetic_router)
v1.include_router(prescription_router)
v1.include_router(drugs_router)
v1.include_router(interactions_router)
v1.include_router(ocr_router)
v1.include_router(history_router)
v1.include_router(ml_router)
v1.include_router(adherence_router)
v1.include_router(admin_router)
app.include_router(v1)

# Backward-compatible un-prefixed routes (deprecated — will be removed in v2)
app.include_router(diabetic_router)
app.include_router(prescription_router)
app.include_router(drugs_router)
app.include_router(interactions_router)
app.include_router(ocr_router)
app.include_router(history_router)
app.include_router(ml_router)
app.include_router(adherence_router)
app.include_router(admin_router)


# ============== Health & Stats Endpoints ==============

@app.get("/", tags=["Health"])
async def root():
    """Root endpoint - API health check."""
    return {
        "message": "Drug-Drug Interaction Prediction API",
        "version": settings.APP_VERSION,
        "status": "healthy",
        "docs": "/docs"
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Detailed health check."""
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": settings.APP_VERSION
    }


@app.get("/health/live", tags=["Health"])
async def liveness():
    """Simple liveness probe."""
    return {"status": "alive"}


@app.get("/health/ready", tags=["Health"])
async def readiness(db: AsyncSession = Depends(get_db)):
    """
    Readiness probe that checks database connectivity and model availability.
    """
    # DB check
    await db.execute(select(func.count(Drug.id)))

    # Model availability check (lightweight)
    model_dir = str(Path(__file__).resolve().parent.parent / "models")
    has_models = all(
        os.path.exists(os.path.join(model_dir, fname))
        for fname in [
            "feature_extractor.pkl",
            "random_forest_model.pkl",
            "xgboost_model.pkl",
            "lightgbm_model.pkl",
        ]
    )

    return {
        "status": "ready" if has_models else "degraded",
        "models_loaded": has_models,
        "request_tracing": True,
        "admin_api_protected": bool(settings.API_KEY),
    }


@app.get("/health/apis", tags=["Health"])
async def api_health_status():
    """
    Get health status of all external APIs used for drug data.
    
    Shows circuit breaker states, cache statistics, and API availability.
    """
    try:
        from app.services.api_client import get_api_client
        
        client = get_api_client()
        status = client.get_health_status()
        
        # Add LLM status
        llm_status = {
            "ollama_configured": bool(settings.OLLAMA_HOST),
            "gemini_configured": bool(settings.GOOGLE_API_KEY),
            "fallback_to_templates": settings.LLM_FALLBACK_TO_TEMPLATES,
        }
        
        # Add API key status (without exposing keys)
        api_keys = {
            "openfda_key_configured": bool(settings.OPENFDA_API_KEY),
            "umls_key_configured": bool(settings.UMLS_API_KEY),
        }
        
        return {
            "status": "healthy",
            "external_apis": status,
            "llm_services": llm_status,
            "api_keys": api_keys,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        
    except Exception as e:
        logger.error(f"API health check error: {e}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

@app.get("/stats", response_model=DatabaseStats, tags=["Statistics"])
async def get_statistics(db: AsyncSession = Depends(get_db)):
    """
    Get database statistics.
    
    Results are cached for 5 minutes to reduce database load.
    """
    # Try cache first
    cached_stats = await get_cached_db_stats()
    if cached_stats is not None:
        logger.debug("Cache HIT for database stats")
        return DatabaseStats(**cached_stats)
    
    logger.debug("Cache MISS for database stats")
    
    # Count drugs
    drug_count = await db.execute(select(func.count(Drug.id)))
    total_drugs = drug_count.scalar()
    
    # Count interactions
    interaction_count = await db.execute(select(func.count(DrugInteraction.id)))
    total_interactions = interaction_count.scalar()
    
    # Count by severity using single GROUP BY query
    severity_query = await db.execute(
        select(DrugInteraction.severity, func.count(DrugInteraction.id))
        .group_by(DrugInteraction.severity)
    )
    severity_results = severity_query.all()
    severity_counts = {row[0]: row[1] for row in severity_results if row[0]}
    
    stats_data = {
        "total_drugs": total_drugs,
        "total_interactions": total_interactions,
        "interactions_by_severity": severity_counts,
        "last_updated": datetime.now(timezone.utc).isoformat(),
    }
    
    # Cache the results
    await cache_db_stats(stats_data)
    
    return DatabaseStats(
        total_drugs=total_drugs,
        total_interactions=total_interactions,
        interactions_by_severity=severity_counts,
        last_updated=datetime.now(timezone.utc)
    )
